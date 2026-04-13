"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import os
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy

# Import table color utilities for background change experiments
from robomimic.scripts.table_color_utils import (
    get_table_color,
    set_table_color_runtime
)



class ObservationKeyMapper(EnvWrapper):

    """Wrapper that remaps observation keys."""
    def __init__(self, env, key_mapping):
        super(ObservationKeyMapper, self).__init__(env)
        self.env = env
        self.key_mapping = key_mapping
    
    def reset(self):
        obs = self.env.reset()
        return self._map_keys(obs)
    
    def reset_to(self, state):
        obs = self.env.reset_to(state)
        return self._map_keys(obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._map_keys(obs), reward, done, info
    
    def _map_keys(self, obs):
        mapped_obs = {}
        for env_key, obs_value in obs.items():
            # print("INFO: env_key:", env_key)
            # print("INFO: self.key_mapping:", self.key_mapping)
            if env_key in self.key_mapping:
                mapped_key = self.key_mapping[env_key]
                mapped_obs[mapped_key] = obs_value
            else:
                mapped_obs[env_key] = obs_value
        return mapped_obs
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class RGBDObservationWrapper(EnvWrapper):
    """
    Merge RGB and depth observations into a single RGBD key.
    Expects RGB in (H, W, 3) and depth in (H, W) or (H, W, 1).
    """
    def __init__(self, env, rgb_key, depth_key, out_key="agentview_image"):
        super(RGBDObservationWrapper, self).__init__(env)
        self.env = env
        self.rgb_key = rgb_key
        self.depth_key = depth_key
        self.out_key = out_key

    def reset(self):
        obs = self.env.reset()
        return self._merge(obs)

    def reset_to(self, state):
        obs = self.env.reset_to(state)
        return self._merge(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._merge(obs), reward, done, info

    def _merge(self, obs):
        if (self.rgb_key not in obs) or (self.depth_key not in obs):
            return obs

        rgb = obs[self.rgb_key]
        depth = obs[self.depth_key]
        if depth is not None and depth.ndim == rgb.ndim - 1:
            depth = depth[..., None]

        if isinstance(rgb, np.ndarray):
            rgb = rgb.astype(np.float32, copy=False)
            depth = depth.astype(np.float32, copy=False)
            rgbd = np.concatenate([rgb, depth], axis=-1)
        else:
            rgb = rgb.float()
            depth = depth.float()
            rgbd = torch.cat([rgb, depth], dim=-1)

        merged = dict(obs)
        merged[self.out_key] = rgbd
        merged.pop(self.depth_key, None)
        return merged

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, table_color_rgb=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        table_color_rgb (list): RGB color for table background change experiment. If provided,
            will modify the table color at runtime.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)
    
    # Apply table color if specified (for background change experiments)
    if table_color_rgb is not None:
        # Get the underlying env if wrapped
        base_env = env.env if isinstance(env, EnvWrapper) else env
        set_table_color_runtime(base_env, table_color_rgb)
    # Debug helper: save obs['agentview_image'] as an image.


    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            act = policy(ob=obs)

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()
            # # Code for saving obs['agentview_image'] as an image.
            # if "agentview_image" in obs:
            #     from PIL import Image
            #     import os

            #     # obs['agentview_image']: [C, H, W] (numpy array or torch tensor)
            #     img = obs["agentview_image"]
            #     # Convert to numpy if this is a torch tensor.
            #     if hasattr(img, "cpu"):
            #         img = img.cpu().numpy()
            #     # [C,H,W] -> [H,W,C]
            #     if img.shape[0] == 3:
            #         img = img.transpose(1, 2, 0)
            #     # Convert from [0, 1] to [0, 255] if needed.
            #     if img.max() <= 1.0:
            #         img = (img * 255).astype("uint8")
            #     else:
            #         img = img.astype("uint8")
                
            #     # Create the output directory.
            #     save_dir = "outputs/debug_obs_img"
            #     os.makedirs(save_dir, exist_ok=True)
            #     img_save_path = os.path.join(save_dir, f"step_{step_i}.png")
                

            #     Image.fromarray(img).save(img_save_path)
            #     print("INFO: saved agentview_image to ", img_save_path)
    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    if args.camera_names and len(args.camera_names) > 0:
        env_meta = ckpt_dict["env_metadata"]
        shape_meta = ckpt_dict["shape_metadata"]
        camera_height = ckpt_dict["env_metadata"]["env_kwargs"]["camera_heights"]
        camera_width = ckpt_dict["env_metadata"]["env_kwargs"]["camera_widths"]
        print(f"Creating environment with camera_names: {args.camera_names}, camera_height: {camera_height}, camera_width: {camera_width}")
        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=args.camera_names,
            camera_height=camera_height,
            camera_width=camera_width,
            reward_shaping=False,
            render=args.render,
            render_offscreen=(args.video_path is not None),
            use_image_obs=shape_meta.get("use_images", False),
            use_depth_obs=(shape_meta.get("use_depths", False) or args.use_rgbd),
        )
        env_obs_key = args.camera_names[0] + "_image"

        # Register both keys as rgb observations.
        ObsUtils.OBS_KEYS_TO_MODALITIES[env_obs_key] = "rgb"
        ObsUtils.OBS_KEYS_TO_MODALITIES['agentview_image'] = "rgb"  # This is the key registration.
        key_mapping = {env_obs_key: 'agentview_image'}
        env = ObservationKeyMapper(env, key_mapping)

        if args.use_rgbd:
            depth_key = args.camera_names[0] + "_depth"
            env = RGBDObservationWrapper(
                env,
                rgb_key="agentview_image",
                depth_key=depth_key,
                out_key="agentview_image",
            )

    else:
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=args.env, 
            render=args.render, 
            render_offscreen=(args.video_path is not None), 
            verbose=True,
        )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Load table color if specified (for background change experiments)
    table_color_rgb = None
    table_color_name = None
    if args.table_color_index is not None:
        color_info = get_table_color(args.table_color_index, args.table_color_config)
        table_color_rgb = color_info["rgb"]
        table_color_name = color_info["name"]
        print(f"Using table color: {table_color_name} (index {args.table_color_index}), RGB: {table_color_rgb}")

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            table_color_rgb=table_color_rgb,
        )
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    parser.add_argument(
        "--use_rgbd",
        action='store_true',
        help="(optional) request depth obs and pack RGB+depth into a 4-channel agentview_image",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    # Table color arguments for background change experiments
    parser.add_argument(
        "--table-color-config",
        type=str,
        default=None,
        help="(optional) path to JSON file with table color definitions. Uses default if not specified.",
    )
    parser.add_argument(
        "--table-color-index",
        type=int,
        default=None,
        help="(optional) index of the table color to use (0-19). If not specified, original color is used.",
    )

    # Manually register mimicgen environments.
    
        
    args = parser.parse_args()
    run_trained_agent(args)
