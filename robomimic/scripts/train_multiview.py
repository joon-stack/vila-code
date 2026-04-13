"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import pickle
import shutil
import psutil
import sys
import socket
import traceback
from datetime import datetime

from collections import OrderedDict
import random

import torch
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.utils.dataset import MultiViewDataset


def train(config, device, resume=False):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, resume=resume)

    # path for latest model and backup (to support @resume functionality)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # For multiview datasets, obs keys in config are expected to be view group names.
    # We expand them to full dataset paths by inspecting the first dataset.
    # This assumes all datasets in this training run have the same view structure.
    first_dataset_path = os.path.expanduser(config.train.data[0]["path"])
    with h5py.File(first_dataset_path, "r") as f:
        demo_id = list(f["data"].keys())[0]
        obs_grp = f[f"data/{demo_id}/obs"]

        with config.values_unlocked():
            all_rgb_keys = []
            for modality_type in config.observation.modalities.obs:
                if modality_type != "rgb":
                    continue

                original_keys = config.observation.modalities.obs[modality_type]
                if not original_keys:
                    continue

                expanded_keys = []
                for view_key in original_keys:
                    if view_key in obs_grp and isinstance(obs_grp[view_key], h5py.Group):
                        found_ds = False
                        # Find the first dataset within the view group
                        for ds_name in obs_grp[view_key]:
                            if isinstance(obs_grp[f"{view_key}/{ds_name}"], h5py.Dataset):
                                expanded_key = f"{view_key}/{ds_name}"
                                expanded_keys.append(expanded_key)
                                # Update ObsUtils mapping as well
                                ObsUtils.OBS_KEYS_TO_MODALITIES[expanded_key] = modality_type
                                found_ds = True
                                break
                        if not found_ds:
                            print(f"Warning: no dataset found in group obs/{view_key} in {first_dataset_path}")
                    else:
                        expanded_keys.append(view_key)
                
                all_rgb_keys.extend(expanded_keys)
                config.observation.modalities.obs[modality_type] = expanded_keys

    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    if isinstance(config.train.data, str):
        # if only a single dataset is provided, convert to list
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

        # populate language instruction for env in env_meta
        env_meta["lang"] = dataset_cfg.get("lang", "dummy")

        # update env meta if applicable
        from robomimic.utils.python_utils import deep_update
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_config=dataset_cfg,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        # if an environment name is specified, just use this env using the first dataset's metadata
        # and ignore envs from all datasets
        env_meta = env_meta_list[0].copy()
        env_meta["env_name"] = config.experiment.env
        env_meta_list = [env_meta]
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        for env_i in range(len(env_meta_list)):
            # check if this env should be evaluated
            dataset_cfg = config.train.data[env_i]
            do_eval = dataset_cfg.get("eval", True)
            if not do_eval:
                continue

            env_meta = env_meta_list[env_i]
            shape_meta = shape_meta_list[env_i]

            env_names = [env_meta["env_name"]]
            if (env_i == 0) and (config.experiment.additional_envs is not None):
                # if additional environments are specified, add them to the list
                # all additional environments use env_meta from the first dataset
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            # create environment for each env_name
            def create_env(env_name):
                env_kwargs = dict(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"],
                )
                env = EnvUtils.create_env_from_metadata(**env_kwargs)
                # handle environment wrappers
                env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable
                return env
            for env_name in env_names:
                env = create_env(env_name)
                env_key = os.path.splitext(os.path.basename(dataset_cfg["path"]))[0] if not dataset_cfg.get("key", None) else dataset_cfg["key"]
                envs[env_key] = env
                print(env)

    print("")

    # Create MultiViewDataset directly, instead of using TrainUtils.load_data_for_training
    # This is because we are using a custom Dataset class.
    # NOTE: This simplified logic assumes a single training dataset.
    # For MetaDataset support, this would need to be expanded.
    
    train_dataset_cfg = config.train.data[0]
    
    # Get image size from config for resize
    img_size = None
    if hasattr(config.observation.encoder.rgb, 'core_kwargs') and 'backbone_kwargs' in config.observation.encoder.rgb.core_kwargs:
        img_size = config.observation.encoder.rgb.core_kwargs['backbone_kwargs'].get('img_size', None)
    
    # Convert img_size to resize_wh tuple if specified
    resize_wh = None
    if img_size is not None:
        if isinstance(img_size, int):
            resize_wh = (img_size, img_size)
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            resize_wh = tuple(img_size)
    
    # Optionally load trajectory split from pickle and select demos
    train_demos = None
    val_demos = None
    traj_split_path = train_dataset_cfg.get("traj_split_path", None)
    if traj_split_path is not None and os.path.exists(os.path.expanduser(traj_split_path)):
        with open(os.path.expanduser(traj_split_path), 'rb') as f:
            split_dict = pickle.load(f)
        train_demos = split_dict.get('train_demo_names', None)
        val_demos = split_dict.get('val_demo_names', None)

    # Get demo_limit from config if specified
    demo_limit = train_dataset_cfg.get("demo_limit", None)
    if train_demos is not None and demo_limit is not None:
        train_demos = train_demos[:demo_limit]

    trainset = MultiViewDataset(
        hdf5_path=train_dataset_cfg["path"],
        obs_keys=config.all_obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        load_next_obs=config.train.hdf5_load_next_obs,
        filter_by_attribute=config.train.hdf5_filter_key,
        demos=train_demos,
        resize_wh=resize_wh,
        demo_limit=demo_limit,
    )

    validset = None
    if config.experiment.validate:
        valid_dataset_cfg = config.train.data[0] # Assuming same dataset, different filter key
        validset = MultiViewDataset(
            hdf5_path=valid_dataset_cfg["path"],
            obs_keys=config.all_obs_keys,
            action_keys=config.train.action_keys,
            dataset_keys=config.train.dataset_keys,
            action_config=config.train.action_config,
            frame_stack=config.train.frame_stack,
            seq_length=config.train.seq_length,
            pad_frame_stack=config.train.pad_frame_stack,
            pad_seq_length=config.train.pad_seq_length,
            get_pad_mask=False,
            goal_mode=config.train.goal_mode,
            hdf5_cache_mode=config.train.hdf5_cache_mode,
            hdf5_use_swmr=config.train.hdf5_use_swmr,
            hdf5_normalize_obs=config.train.hdf5_normalize_obs,
            load_next_obs=config.train.hdf5_load_next_obs,
            filter_by_attribute=config.train.hdf5_validation_filter_key,
            demos=val_demos,
            resize_wh=resize_wh,
        )

    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # Since MultiViewDataset wraps the original dataset, we modify the config *after*
    # the dataset has been created with the multi-view keys.
    canonical_view_key = "agentview_image"
    
    # Save original config for later saving (before modifying it)
    original_config = config.dump()
    
    # Update config and shape_meta to reflect the single canonical view for the model
    with config.values_unlocked():
        # Set the rgb modality to only have the new canonical key
        config.observation.modalities.obs["rgb"] = [canonical_view_key]

    # Modify shape_meta: remove old view keys and add the new canonical key
    if len(all_rgb_keys) > 0:
        # Assume all views have the same shape, take the first one
        original_view_shape = shape_meta_list[0]["all_shapes"][all_rgb_keys[0]]
        
        # Update shape if resize is applied
        if resize_wh is not None:
            # Update the shape to reflect the resized dimensions
            if len(original_view_shape) == 3:  # (C, H, W)
                new_shape = [original_view_shape[0], resize_wh[1], resize_wh[0]]  # (C, new_H, new_W)
            elif len(original_view_shape) == 4:  # (T, C, H, W)
                new_shape = [original_view_shape[0], original_view_shape[1], resize_wh[1], resize_wh[0]]  # (T, C, new_H, new_W)
            else:
                new_shape = original_view_shape
            # print(f"DEBUG: Updating shape from {original_view_shape} to {new_shape}")
        else:
            new_shape = original_view_shape
            
        for meta in shape_meta_list:
            for key in all_rgb_keys:
                if key in meta["all_shapes"]:
                    del meta["all_shapes"][key]
            meta["all_shapes"][canonical_view_key] = new_shape

    # Update ObsUtils mapping for the new canonical key BEFORE model creation.
    ObsUtils.OBS_KEYS_TO_MODALITIES[canonical_view_key] = 'rgb'

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # maybe retreve statistics for normalizing actions
    action_normalization_stats = trainset.get_action_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # add info to optim_params
    with config.values_unlocked():
        if "optim_params" in config.algo:
            # add info to optim_params of each net
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs
        # handling for "hbc" and "iris" algorithms
        if config.algo_name == "hbc":
            for sub_algo in ["planner", "actor"]:
                # add info to optim_params of each net
                for k in config.algo[sub_algo].optim_params:
                    config.algo[sub_algo].optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                    config.algo[sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs
        if config.algo_name == "iris":
            for sub_algo in ["planner", "value"]:
                # add info to optim_params of each net
                for k in config.algo["value_planner"][sub_algo].optim_params:
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device
    )


    if resume:
        # load ckpt dict
        print("*" * 50)
        print("resuming from ckpt at {}".format(latest_model_path))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_path)
        except Exception as e:
            print("got error: {} when loading from {}".format(e, latest_model_path))
            print("trying backup path {}".format(latest_model_backup_path))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_backup_path)
        # load model weights and optimizer state # TODO: revert this
        model.deserialize(ckpt_dict["model"], load_optimizers=False)
        print("*" * 50)
    
    # if checkpoint is specified, load in model weights;
    # will not use ckpt_path if resuming training
    ckpt_path = config.experiment.ckpt_path
    if (ckpt_path is not None) and (not resume):
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])
    obs_encoder_path = config.experiment.obs_encoder_path
    obs_encoder_freeze = config.experiment.obs_encoder_freeze
    if obs_encoder_path is not None and (not resume):
        print("LOADING OBS ENCODER FROM " + obs_encoder_path)
        obs_encoder = model.nets['policy']["obs_encoder"]
        state_dict = torch.load(obs_encoder_path)
        _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=obs_encoder_path, device=device, verbose=True)
        ckpt_dict = ckpt_dict['model']['nets']
        # print("INFO: ckpt_dict['model']['nets'].keys(): " + str(ckpt_dict['model']['nets'].keys()))
        # print("INFO: ckpt_dict['model']['nets']['policy']['obs_encoder'].keys(): " + str(ckpt_dict['model']['nets']['policy']['obs_encoder'].keys()))
        # Extract keys with the policy.obs_encoder. prefix.
        obs_encoder_prefix = "policy.obs_encoder."
        obs_encoder_keys = {}
        for key in ckpt_dict:
            if key.startswith(obs_encoder_prefix):
                clean_key = key.replace(obs_encoder_prefix, "")
                obs_encoder_keys[clean_key] = ckpt_dict[key]
        obs_encoder.load_state_dict(obs_encoder_keys, strict=True)
        model.nets['policy']['obs_encoder'] = obs_encoder
        
        print("INFO: obs_encoder successfully loaded from " + obs_encoder_path)

        if obs_encoder_freeze:
            for param in model.nets['policy']['obs_encoder'].parameters():
                param.requires_grad = False
            model.nets['policy']['obs_encoder'].eval()
        else:
            obs_encoder = model.nets['policy']['obs_encoder']
            obs_encoder_params = set(obs_encoder.parameters())
            
            # Update the existing optimizer param_groups.
            if 'policy' in model.optimizers:
                optimizer = model.optimizers['policy']
                
                # Read the initial learning rate from the config.
                policy_optim_params = model.optim_params.get('policy', None)
                if policy_optim_params is None:
                    # Try an alternate config path.
                    if hasattr(model, 'algo_config') and hasattr(model.algo_config, 'optim_params'):
                        policy_optim_params = model.algo_config.optim_params.get('policy', None)
                
                base_lr = policy_optim_params["learning_rate"]["initial"] if policy_optim_params else optimizer.param_groups[0].get('lr', 1e-4)
                
                # Copy all settings from the existing group (betas, eps, amsgrad, etc.).
                existing_group = optimizer.param_groups[0]
                obs_encoder_group = {k: v for k, v in existing_group.items() if k != 'params'}  # Copy everything except params.
                obs_encoder_group['params'] = list(obs_encoder_params)  # Set the new params.
                obs_encoder_group['lr'] = base_lr * 0.1
                print("INFO: base_lr: ", base_lr, "obs_encoder_lr: ", obs_encoder_group['lr'])
                
                # Remove obs_encoder parameters from the original group.
                other_params = [p for p in existing_group['params'] if p not in obs_encoder_params]
                optimizer.param_groups[0]['params'] = other_params
                
                # Add the dedicated obs_encoder group.
                optimizer.param_groups.append(obs_encoder_group)
    if obs_encoder_freeze:
        for param in model.nets['policy']['obs_encoder'].parameters():
            param.requires_grad = False
        model.nets['policy']['obs_encoder'].eval()
        print("INFO: obs_encoder successfully frozen")

    
        
        

    # save the original config as a json file (before multiview modifications)
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(original_config, outfile, indent=4)
    
    # also save original config in checkpoint directory for easy access
    if ckpt_dir is not None:
        with open(os.path.join(ckpt_dir, 'config.json'), 'w') as outfile:
            json.dump(original_config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    start_epoch = 1 # epoch numbers start at 1
    if resume:
        # load variable state needed for train loop
        variable_state = ckpt_dict["variable_state"]
        start_epoch = variable_state["epoch"] + 1 # start at next epoch, since this recorded the last epoch of training completed
        best_valid_loss = variable_state["best_valid_loss"]
        best_return = variable_state["best_return"]
        best_success_rate = variable_state["best_success_rate"]
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)

    for epoch in tqdm(range(start_epoch, config.train.num_epochs + 1), desc="Training Epochs"):
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:
            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # get variable state for saving model
        variable_state = dict(
            epoch=epoch,
            best_valid_loss=best_valid_loss,
            best_return=best_return,
            best_success_rate=best_success_rate,
        )

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:    
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list)==1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list)==1 else shape_meta_list,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

            # also save latest model for resume functionality
            print("\nsaving latest model at {}...\n".format(latest_model_path))
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list)==1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list)==1 else shape_meta_list,
                variable_state=variable_state,
                ckpt_path=latest_model_path,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

            # keep a backup model in case last.pth is malformed (e.g. job died last time during saving)
            shutil.copyfile(latest_model_path, latest_model_backup_path)
            print("\nsaved backup of latest model at {}\n".format(latest_model_backup_path))

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = [{"path": args.dataset}]

    if args.name is not None:
        config.experiment.name = args.name
    else:
        # if no name is provided, use config filename (without .json extension)
        if args.config is not None:
            config_filename = os.path.basename(args.config)
            config.experiment.name = os.path.splitext(config_filename)[0]
        else:
            # fallback to datetime if no config file
            config.experiment.name = datetime.now().strftime("%Y%m%d%H%M%S")

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device, resume=args.resume)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # resume training from latest checkpoint
    parser.add_argument(
        "--resume",
        action='store_true',
        help="set this flag to resume training from latest checkpoint",
    )

    args = parser.parse_args()
    main(args)
