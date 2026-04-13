import math
import os
import random
import shutil
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, List, Any
import yaml

import numpy as np
import pyrallis
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import wandb
from pyrallis import field
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from functools import partial

from vila_stage1.augmentations import Augmenter
from vila_stage1.nn import Actor, LAOMWithLabels
from vila_stage1.scheduler import linear_annealing_with_warmup
from vila_stage1.utils import (
    DCSMVInMemoryDataset,
    get_grad_norm,
    get_optim_groups,
    normalize_img,
    set_seed,
    soft_update,
    metric_learning_collate_fn,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_or_create_traj_splits(hdf5_path, train_traj_num=None, val_traj_num=None, seed=0, dataset_type="unlabeled"):
    """
    Compute fixed train / validation demo-name splits based on
    HDF5 `data/<demo>` trajectory ids.

    Returns:
        (train_demo_names: List[str], val_demo_names: List[str])  # val is [] when no validation split is used
    """
    with h5py.File(hdf5_path, 'r') as f:
        all_demos = sorted(list(f['data'].keys()))

    total = len(all_demos)
    if dataset_type == "unlabeled":
        if train_traj_num is None and val_traj_num is None:
            # Use the entire set for training.
            train_traj_num = total
            val_traj_num = 0
        elif train_traj_num is None:
            train_traj_num = max(0, total - (val_traj_num or 0))
        elif val_traj_num is None:
            val_traj_num = max(0, total - train_traj_num)
    elif dataset_type == "labeled":
        if train_traj_num is None and val_traj_num is None:
            train_traj_num = int(0.8 * total)
            val_traj_num = total - train_traj_num
        elif train_traj_num is None:
            train_traj_num = max(0, total - (val_traj_num or 0))
        elif val_traj_num is None:
            val_traj_num = max(0, total - train_traj_num)

    if train_traj_num + val_traj_num > total:
        val_traj_num = max(0, total - train_traj_num)

    g = np.random.default_rng(seed)
    perm = g.permutation(total)
    train_idx = perm[:train_traj_num]
    val_idx = perm[train_traj_num:train_traj_num + val_traj_num]
    train_demo_names = [all_demos[i] for i in sorted(train_idx)]
    val_demo_names = [all_demos[i] for i in sorted(val_idx)]

    head_n = min(5, len(train_demo_names))
    if head_n > 0:
        print(f"Train demos head ({head_n}): {train_demo_names[:head_n]}")
    head_nv = min(5, len(val_demo_names))
    if head_nv > 0:
        print(f"Val demos head ({head_nv}): {val_demo_names[:head_nv]}")

    return train_demo_names, val_demo_names


def compute_gram_loss(X, Y):
    """
    Compute Gram matrix loss between latent and true action representations.
    
    Args:
        X: [batch_size, latent_dim] - latent action (will be L2 normalized)
        Y: [batch_size, seq_len * act_dim] - true action sequences

    Returns:
        MSE loss between normalized Gram matrices (lower is better)
    
    Gram matrix represents pairwise similarity between samples in a batch.
    By matching Gram matrices of latent and true action spaces, we align
    their global similarity structure without requiring kernel bandwidth tuning.
    """
    # Guard against small batches
    n = X.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=X.device, requires_grad=True)
    
    X_norm = F.normalize(X, p=2, dim=1)
    # Compute pairwise cosine distance: 1 - cosine_similarity
    cosine_sim_X = torch.matmul(X_norm, X_norm.T)  # [B, B] cosine similarity matrix
    G_X = 1 - cosine_sim_X  # Convert to cosine distance
    G_X = F.normalize(G_X, p=2, dim=1)  # L2 normalize each row
    Y_norm = F.normalize(Y, p=2, dim=1)
    cosine_sim_Y = torch.matmul(Y_norm, Y_norm.T)
    G_Y = 1 - cosine_sim_Y  # Convert to cosine distance
    G_Y = F.normalize(G_Y, p=2, dim=1)  # L2 normalize each row
    
    # MSE loss between the two Gram matrices
    loss = F.mse_loss(G_X, G_Y)
    
    return loss


@dataclass
class LAOMConfig:
    device: str = "cuda"
    num_epochs: int = 100
    batch_size: int = 32
    camera_num: int = 8
    view_keys: List[Any] = field(default_factory=list)
    views_per_instance: int = 4  # K: views to sample per instance
    contrastive_loss_coef: float = 1.0
    labeled_loss_coef: float = 0.05
    cosine_loss: bool = False
    use_aug: bool = False
    future_obs_offset: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = 1.0
    latent_action_dim: int = 256
    act_head_dim: int = 512
    act_head_dropout: float = 0.0
    obs_head_dim: int = 512
    obs_head_dropout: float = 0.0
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 1
    encoder_dropout: float = 0.0
    encoder_norm_out: bool = True
    encoder_deep: bool = True
    target_tau: float = 0.01
    target_update_every: int = 1
    frame_stack: int = 3
    data_path: str = "data/test.hdf5"
    seed: int = 0
    infonce_temperature: float = 0.1
    weighted_infonce_beta: float = 1.0
    use_random_offset: bool = True  # If False, always use future_obs_offset; otherwise sample uniformly from [1, future_obs_offset].
    alignment_loss_coef: float = 1.0  # Gram / global similarity alignment coefficient

    mixed_view_sampling: bool = False  # Sample different views per positive sample
    positive_samples_per_instance: int = 4  # Number of positive samples per instance when using mixed view sampling
    resize_size: int = 64  # Image resize size

    # Trajectory-level splits (number of demos)
    train_traj_num: Optional[int] = None
    val_traj_num: Optional[int] = None
    unlabeled_train_traj_num: Optional[int] = None
    unlabeled_val_traj_num: Optional[int] = None

    # For SO101
    action_key: str = "actions" # "joints", "delta_joints", "ee", "delta_ee", "actions (unknown)"
    filter_noop: bool = False # If True, zero action is filtered out

    num_trajectories_to_load: int = 200

    def __post_init__(self):
        if not self.view_keys:
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in range(self.camera_num)]
        elif self.view_keys and isinstance(self.view_keys[0], int):
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in self.view_keys]


@dataclass
class BCConfig:
    num_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 2
    encoder_deep: bool = False
    dropout: float = 0.0
    use_aug: bool = True
    frame_stack: int = 3
    data_path: str = "data/test.hdf5"
    eval_seed: int = 0
    seed: int = 0  # Random seed for BC training.
    # Multi-view related settings
    camera_num: int = 20
    view_keys: List[Any] = field(default_factory=list)
    views_per_instance: int = 1  # K: views to sample per instance
    fixed_offset: int = 1  # Fixed offset for all samples

    # Trajectory-level splits (number of demos)
    train_traj_num: Optional[int] = None
    val_traj_num: Optional[int] = None
    
    # Image resize
    resize_size: int = 64
    
    # View sampling configuration
    mixed_view_sampling: bool = False
    positive_samples_per_instance: int = 4

    # For SO101
    action_key: str = "actions" # "joints", "delta_joints", "ee", "delta_ee", "actions (unknown)"
    filter_noop: bool = False # If True, zero action is filtered out

    num_trajectories_to_load: int = 200
    def __post_init__(self):
        if not self.view_keys:
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in range(self.camera_num)]
        elif self.view_keys and isinstance(self.view_keys[0], int):
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in self.view_keys]


@dataclass
class Config:
    project: str = "vila"
    group: str = "stage1"
    name: str = "vila-stage1"
    run_name: Optional[str] = None  # Custom run name for checkpoint dir and wandb
    output_root: str = "outputs/stage1"
    seed: int = 0
    lapo_checkpoint_path: Optional[str] = None
    bc_checkpoint_path: Optional[str] = None

    lapo: LAOMConfig = field(default_factory=LAOMConfig)
    bc: BCConfig = field(default_factory=BCConfig)

    def __post_init__(self):
        self.lapo.seed = self.seed
        # Keep the BC seed aligned with the top-level config seed.
        self.bc.seed = self.seed
        # Keep BC on the same camera subset as VILA pretraining unless explicitly overridden.
        if not self.bc.view_keys:
            self.bc.view_keys = list(self.lapo.view_keys)
        self.bc.camera_num = len(self.bc.view_keys)


def train_mv(config: LAOMConfig, checkpoint_dir: str, config_path: str = None):
    set_seed(config.seed)
    DEVICE = config.device

    # --- Print view sampling configuration ---
    print("--- View Sampling Configuration ---")
    print(f"Mixed view sampling: {config.mixed_view_sampling}")
    if config.mixed_view_sampling:
        print(f"Views per instance (K): {config.views_per_instance}")
        print(f"Positive samples per instance: {config.positive_samples_per_instance}")
        print(f"Available views: {config.view_keys}")
    print("-----------------------------")
    
    # --- Print loss configuration ---
    print("--- Loss Configuration ---")
    print(f"Contrastive loss coefficient: {config.contrastive_loss_coef}")
    print(f"Weighted InfoNCE temperature: {config.infonce_temperature}")
    print(f"Weighted InfoNCE beta: {config.weighted_infonce_beta}")
    print("Distance type: l2")
    print(f"Gram alignment coefficient: {config.alignment_loss_coef}")
    print("-----------------------------")


    # Stage-1: unlabeled traj splits
    unl_train_demos, _ = get_or_create_traj_splits(
        hdf5_path=config.data_path,
        train_traj_num=config.unlabeled_train_traj_num,
        val_traj_num=config.unlabeled_val_traj_num,
        seed=config.seed,
        dataset_type="unlabeled",
    )
    dataset = DCSMVInMemoryDataset(
        config.data_path,
        max_offset=config.future_obs_offset,
        frame_stack=config.frame_stack,
        device="cpu",
        camera_num=config.camera_num,
        view_keys=config.view_keys,
        resize_wh=(config.resize_size, config.resize_size),
        selected_demo_names=unl_train_demos,
        mixed_view_sampling=config.mixed_view_sampling,
        positive_samples_per_instance=config.positive_samples_per_instance,
        action_key=config.action_key,
        filter_noop=config.filter_noop,
        num_trajectories_to_load=config.num_trajectories_to_load,
    )

    # Use trajectory-based loading directly instead of index-based splitting.
    train_dataset = dataset

    random_offset = config.future_obs_offset if not config.use_random_offset else random.randint(1, config.future_obs_offset)
    train_dataset.set_fixed_offset(random_offset)
    print(f"[Dynamic Offset] Initial offset = {random_offset} (use_random_offset={config.use_random_offset})")

    collate_fn = partial(metric_learning_collate_fn, K=config.views_per_instance, mixed_view_sampling=config.mixed_view_sampling)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    lapo = LAOMWithLabels(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        true_act_dim=dataset.act_dim,
        latent_act_dim=config.latent_action_dim,
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        obs_head_dim=config.obs_head_dim,
        obs_head_dropout=config.obs_head_dropout,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        encoder_dropout=config.encoder_dropout,
        encoder_norm_out=config.encoder_norm_out,
    ).to(DEVICE)

    target_lapo = deepcopy(lapo)
    for p in target_lapo.parameters():
        p.requires_grad_(False)

    torchinfo.summary(
        lapo,
        input_size=[
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        ],
    )
    optim = torch.optim.Adam(
        params=get_optim_groups(lapo, config.weight_decay),
        lr=config.learning_rate,
        fused=True,
    )
    augmenter = Augmenter(dataset.img_hw)

    state_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.state_dim).to(DEVICE)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)

    # act_linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    act_linear_probe = nn.Sequential(
        nn.Linear(config.latent_action_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, dataset.act_dim)
    ).to(DEVICE)
    act_probe_optim = torch.optim.Adam(act_linear_probe.parameters(), lr=config.learning_rate)

    print("Final encoder shape:", math.prod(lapo.final_encoder_shape))
    state_act_linear_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.act_dim).to(DEVICE)
    state_act_probe_optim = torch.optim.Adam(state_act_linear_probe.parameters(), lr=config.learning_rate)

    total_updates = len(train_dataloader) * config.num_epochs
    warmup_updates = len(train_dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    start_time = time.time()
    total_iterations = 0
    total_tokens = 0

    base_contrastive_loss_coef = config.contrastive_loss_coef
    for epoch in trange(config.num_epochs, desc="Epochs"):
        # Contrastive loss coefficient warm-up

        current_contrastive_coef = base_contrastive_loss_coef

        lapo.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False):
            if batch is None: continue
            
            random_offset = config.future_obs_offset if not config.use_random_offset else random.randint(1, config.future_obs_offset)
            train_dataset.set_fixed_offset(random_offset)
            
            total_tokens += config.batch_size * config.views_per_instance
            total_iterations += 1

            # action_sequences_mask is added
            obs, next_obs, future_obs, debug_actions, debug_action_sequences, debug_states, instance_ids, offsets, action_sequences_mask = [
                batch[k].to(DEVICE) for k in ["obs", "next_obs", "future_obs", "actions", "action_sequences", "states", "instance_ids", "offsets", "action_sequences_mask"]
            ]

            # Convert flattened multiview batches to channel-first tensors.
            obs = obs.permute((0, 3, 1, 2))
            next_obs = next_obs.permute((0, 3, 1, 2))
            future_obs = future_obs.permute((0, 3, 1, 2))
            
            # Normalize image observations before encoding.
            obs = normalize_img(obs)
            next_obs = normalize_img(next_obs)
            future_obs = normalize_img(future_obs)

            if config.use_aug:
                obs_to_use = augmenter(obs)
                next_obs_to_use = augmenter(next_obs)
                future_obs_to_use = augmenter(future_obs)
            else:
                obs_to_use = obs
                next_obs_to_use = next_obs
                future_obs_to_use = future_obs

            # update lapo
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                latent_next_obs, latent_action, pred_action, obs_hidden = lapo(
                    obs_to_use, 
                    future_obs_to_use, 
                    predict_true_act=True
                )

                

                with torch.no_grad():
                    # Use future_obs_to_use as the target representation.
                    future_obs_target = target_lapo.encoder(future_obs_to_use).flatten(1)

                if config.cosine_loss:
                    loss0 = 1 - F.cosine_similarity(latent_next_obs, future_obs_target.detach(), dim=-1).mean()
                else:
                    loss0 = F.mse_loss(latent_next_obs, future_obs_target.detach())

                # loss with true actions
                loss1 = F.mse_loss(pred_action, debug_actions)

            z_a = latent_action
            z_a_std = torch.std(latent_action, dim=0).mean().item()

            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                sim_matrix = torch.matmul(z_a, z_a.T) / config.infonce_temperature
                sim_matrix.fill_diagonal_(-1e9)

                sim_matrix_max = torch.max(sim_matrix, dim=1, keepdim=True)[0]
                sim_matrix_stable = sim_matrix - sim_matrix_max
                log_sum_exp = torch.log(torch.sum(torch.exp(sim_matrix_stable), dim=1, keepdim=True)) + sim_matrix_max
                log_prob = sim_matrix - log_sum_exp

                with torch.no_grad():
                    action_seqs = debug_action_sequences
                    dist_actions_sq = torch.cdist(action_seqs, action_seqs, p=2.0) ** 2
                    weights = F.softmax(-dist_actions_sq / config.weighted_infonce_beta, dim=1)
                    weights.fill_diagonal_(0)

                loss_contrastive = -torch.sum(weights * log_prob) / z_a.shape[0]
                loss_gram = compute_gram_loss(latent_action, debug_action_sequences)

            loss = (
                loss0
                + config.labeled_loss_coef * loss1
                + current_contrastive_coef * loss_contrastive
                + config.alignment_loss_coef * loss_gram
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(lapo.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()
            if i % config.target_update_every == 0:
                soft_update(target_lapo, lapo, tau=config.target_tau)

            # update state probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_hidden.detach())
                state_probe_loss = F.mse_loss(pred_states, debug_states)

            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()

            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action_probe = act_linear_probe(latent_action.detach())
                act_probe_loss = F.mse_loss(pred_action_probe, debug_actions)

            act_probe_optim.zero_grad(set_to_none=True)
            act_probe_loss.backward()
            act_probe_optim.step()

            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                state_pred_action = state_act_linear_probe(obs_hidden.detach())
                state_act_probe_loss = F.mse_loss(state_pred_action, debug_actions)

            state_act_probe_optim.zero_grad(set_to_none=True)
            state_act_probe_loss.backward()
            state_act_probe_optim.step()

            # Prepare logging data
            log_data = {
                "lapo/total_loss": loss.item(),
                "lapo/mse_loss": loss0.item(),
                "lapo/true_action_mse_loss": loss1.item(),
                "lapo/contrastive_loss": loss_contrastive.item(),
                "lapo/state_probe_mse_loss": state_probe_loss.item(),
                "lapo/action_probe_mse_loss": act_probe_loss.item(),
                "lapo/state_action_probe_mse_loss": state_act_probe_loss.item(),
                "lapo/throughput": total_tokens / (time.time() - start_time),
                "lapo/learning_rate": scheduler.get_last_lr()[0],
                "lapo/grad_norm": get_grad_norm(lapo).item(),
                "lapo/target_obs_norm": torch.norm(future_obs_target, p=2, dim=-1).mean().item(),
                "lapo/online_obs_norm": torch.norm(latent_next_obs, p=2, dim=-1).mean().item(),
                "lapo/latent_act_norm": torch.norm(latent_action, p=2, dim=-1).mean().item(),
                "lapo/epoch": epoch,
                "lapo/total_steps": total_iterations,
                "lapo/latent_act_std": z_a_std,
                "lapo/gram_loss": loss_gram.item(),
            }
            
            log_data["lapo/infonce_temperature"] = config.infonce_temperature
            log_data["lapo/contrastive_loss_coef"] = current_contrastive_coef
            wandb.log(log_data)
            

        
        
        if (epoch + 1) % 10000 == 0:
            save_checkpoint(
                lapo,
                optim,
                scheduler,
                epoch,
                loss.item(),
                os.path.join(checkpoint_dir, f"idm_epoch_{epoch+1}.pt"),
                config,
                config_path,
            )

    save_checkpoint(
        lapo,
        optim,
        scheduler,
        config.num_epochs - 1,
        loss.item(),
        os.path.join(checkpoint_dir, "idm.pt"),
        config,
        config_path,
    )

    return lapo


def convert_tuples_to_lists(obj):
    """Recursively convert tuples inside dicts/lists into lists."""
    if isinstance(obj, dict):
        return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tuples_to_lists(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return obj


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, config=None, config_path=None):
    """Save a model checkpoint."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    # Save the config as a separate YAML file when available.
    if config is not None and config_path is not None:
        config_filepath = filepath.replace('.pt', '_config.yaml')
        # Copy the original config file as-is.
        shutil.copy2(config_path, config_filepath)
        print(f"Saved config to: {config_filepath}")
    elif config is not None:
        # Fall back to serializing the in-memory config if config_path is unavailable.
        config_filepath = filepath.replace('.pt', '_config.yaml')
        config_dict = convert_tuples_to_lists(asdict(config))
        with open(config_filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        print(f"Saved config to: {config_filepath}")
    
    torch.save(checkpoint_data, filepath)
    print(f"Saved checkpoint to: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath):
    """Load a model checkpoint."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        # Debugging: print shapes before loading
        if 'model_state_dict' in checkpoint and hasattr(model, 'true_actions_head'):
            print("--- Shape Debug ---")
            # Shape in the current model
            current_shape = model.true_actions_head.weight.shape
            print(f"Current model's 'true_actions_head' shape: {current_shape}")
            
            # Shape in the checkpoint
            checkpoint_shape = checkpoint['model_state_dict']['true_actions_head.weight'].shape
            print(f"Checkpoint's 'true_actions_head' shape: {checkpoint_shape}")
            print("-------------------")

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint: {filepath} (epoch {checkpoint['epoch']})")
        
        # Load and print the config YAML if present.
        config_filepath = filepath.replace('.pt', '_config.yaml')
        if os.path.exists(config_filepath):
            with open(config_filepath, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded config file: {config_filepath}")
            print("Config stored with the checkpoint:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            return checkpoint['epoch'], checkpoint['loss'], config
        
        return checkpoint['epoch'], checkpoint['loss'], None
    return 0, float('inf'), None


def train_bc(lam: LAOMWithLabels, config: BCConfig, checkpoint_dir: str, config_path: str = None):
    # Trajectory-level splits for BC stage
    
    bc_train_demos, _ = get_or_create_traj_splits(
        hdf5_path=config.data_path,
        train_traj_num=config.train_traj_num,
        val_traj_num=config.val_traj_num,
        seed=config.seed,
        dataset_type="unlabeled",
    )

    # BC stage uses a fixed view set and offset.
    train_dataset = DCSMVInMemoryDataset(
        config.data_path,
        max_offset=config.fixed_offset,
        frame_stack=config.frame_stack,
        device="cpu",
        camera_num=config.camera_num,
        view_keys=config.view_keys,
        resize_wh=(config.resize_size, config.resize_size),
        selected_demo_names=bc_train_demos,
        mixed_view_sampling=False,
        positive_samples_per_instance=4,
        action_key=config.action_key,
        filter_noop=config.filter_noop,
        num_trajectories_to_load=config.num_trajectories_to_load,
    )

    collate_fn = partial(metric_learning_collate_fn, K=config.views_per_instance, mixed_view_sampling=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Use a fixed temporal offset during BC.
    train_dataset.set_fixed_offset(config.fixed_offset)

    print(f"\n=== train_bc Dataset Debug ===")
    print(f"Requested fixed offset: {config.fixed_offset}")
    print(f"Dataset has set_fixed_offset method: {hasattr(train_dataset, 'set_fixed_offset')}")
    if hasattr(train_dataset, 'fixed_offset'):
        print(f"Train dataset fixed_offset: {train_dataset.fixed_offset}")
    print("=" * 40)

    num_actions = lam.latent_act_dim
    for p in lam.parameters():
        p.requires_grad_(False)
    lam.eval()

    actor = Actor(
        shape=(3 * config.frame_stack, train_dataset.img_hw, train_dataset.img_hw),
        num_actions=num_actions,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        dropout=config.dropout,
    ).to(DEVICE)

    optim = torch.optim.AdamW(params=get_optim_groups(actor, config.weight_decay), lr=config.learning_rate, fused=True)
    total_updates = len(train_dataloader) * config.num_epochs
    warmup_updates = len(train_dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    print("Latent action dim:", num_actions)
    # Predict a flattened fixed-offset action sequence.
    output_dim = config.fixed_offset * train_dataset.act_dim
    print("Output dim:", output_dim)
    
    print(f"\n=== train_bc Output Dim Debug ===")
    print(f"Fixed offset: {config.fixed_offset}")
    print(f"Dataset act_dim: {train_dataset.act_dim}")
    print(f"Calculated output_dim: {output_dim}")
    print(f"Expected sequence length: {config.fixed_offset}")
    print(f"Expected flattened length: {(config.fixed_offset) * train_dataset.act_dim}")
    print(f"Match with expected: {output_dim == (config.fixed_offset) * train_dataset.act_dim}")
    print("=" * 50)
    
    act_decoder = nn.Sequential(
        nn.Linear(num_actions, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, output_dim)
    ).to(DEVICE)

    act_decoder_optim = torch.optim.AdamW(params=act_decoder.parameters(), lr=config.learning_rate, fused=True)
    act_decoder_scheduler = linear_annealing_with_warmup(act_decoder_optim, warmup_updates, total_updates)

    torchinfo.summary(actor, input_size=(1, 3 * config.frame_stack, train_dataset.img_hw, train_dataset.img_hw))
    if config.use_aug:
        augmenter = Augmenter(img_resolution=train_dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        actor.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False):
            total_tokens += config.batch_size
            total_steps += 1

            # Convert flattened multiview batches to channel-first tensors.
            obs, next_obs, future_obs, debug_actions, debug_action_sequences, debug_states, instance_ids, offsets, action_sequences_mask = [
                batch[k].to(DEVICE) for k in ["obs", "next_obs", "future_obs", "actions", "action_sequences", "states", "instance_ids", "offsets", "action_sequences_mask"]
            ]

            obs = obs.permute((0, 3, 1, 2))
            next_obs = next_obs.permute((0, 3, 1, 2))
            future_obs = future_obs.permute((0, 3, 1, 2))
            
            # Normalize image observations before encoding.
            obs = normalize_img(obs)
            next_obs = normalize_img(next_obs)
            future_obs = normalize_img(future_obs)

            # Label observations with the frozen stage-1 latent actions.
            target_actions = lam.label(obs, future_obs)

            # Apply augmentation only to the BC inputs.
            if config.use_aug:
                obs = augmenter(obs)

            # Update the latent BC policy.
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_actions, _ = actor(obs)
                loss = F.mse_loss(pred_actions, target_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            # Train a probe that maps latent actions back to action sequences.
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_true_actions = act_decoder(pred_actions.detach())
                # Use action_sequences for fixed offset
                decoder_loss = F.mse_loss(pred_true_actions, debug_action_sequences)

            act_decoder_optim.zero_grad(set_to_none=True)
            decoder_loss.backward()
            act_decoder_optim.step()
            act_decoder_scheduler.step()

            wandb.log(
                {
                    "bc/mse_loss": loss.item(),
                    "bc/throughput": total_tokens / (time.time() - start_time),
                    "bc/learning_rate": scheduler.get_last_lr()[0],
                    "bc/act_decoder_probe_mse_loss": decoder_loss.item(),
                    "bc/epoch": epoch,
                    "bc/total_steps": total_steps,
                }
            )
        

        if (epoch + 1) % 1000 == 0:
            save_checkpoint(
                actor,
                optim,
                scheduler,
                epoch,
                loss.item(),
                os.path.join(checkpoint_dir, f"bc_epoch_{epoch+1}.pt"),
                config,
                config_path,
            )

    save_checkpoint(
        actor,
        optim,
        scheduler,
        config.num_epochs - 1,
        loss.item(),
        os.path.join(checkpoint_dir, "bc.pt"),
        config,
        config_path,
    )

    return actor


@pyrallis.wrap()
def train(config: Config, config_path: str = None):
    set_seed(config.seed)

    # Configure the checkpoint directory and wandb run name.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.lapo_checkpoint_path:
        checkpoint_dir = os.path.dirname(config.lapo_checkpoint_path)
        run_name = config.run_name if config.run_name else os.path.basename(checkpoint_dir)
    elif config.bc_checkpoint_path:
        checkpoint_dir = os.path.dirname(config.bc_checkpoint_path)
        run_name = config.run_name if config.run_name else os.path.basename(checkpoint_dir)
    else:
        if config.run_name:
            checkpoint_dir = os.path.join(config.output_root, config.run_name)
            run_name = config.run_name
        else:
            checkpoint_dir = os.path.join(config.output_root, timestamp)
            run_name = timestamp
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Run name: {run_name}")


    run = wandb.init(
        project=config.project,
        group=config.group,
        name=run_name,
        config=asdict(config),
        save_code=True,
    )

    # Save the run config immediately.
    try:
        run_config_path = os.path.join(checkpoint_dir, "config.yaml")
        if config_path is not None and os.path.exists(config_path):
            shutil.copy2(config_path, run_config_path)
        else:
            config_dict = convert_tuples_to_lists(asdict(config))
            with open(run_config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        print(f"Saved run config to: {run_config_path}")
    except Exception as e:
        print(f"Failed to save run config: {e}")

    # Stage 1: VILA pretraining on unlabeled multiview data.
    if config.lapo_checkpoint_path:
        print("=== Stage 1: LAPO Pretraining (skipped, loading from checkpoint) ===")
        # Recreate the encoder using dataset metadata before loading the checkpoint.
        dataset = DCSMVInMemoryDataset(
            config.lapo.data_path, 
            max_offset=config.lapo.future_obs_offset, 
            frame_stack=config.lapo.frame_stack, 
            device="cpu",
            camera_num=config.lapo.camera_num,
            view_keys=config.lapo.view_keys,
            resize_wh=(config.lapo.resize_size, config.lapo.resize_size),
            mixed_view_sampling=config.lapo.mixed_view_sampling,
            positive_samples_per_instance=config.lapo.positive_samples_per_instance,
            action_key=config.lapo.action_key,
            filter_noop=config.lapo.filter_noop,
            num_trajectories_to_load=1,
        )
        lapo = LAOMWithLabels(
            shape=(3 * config.lapo.frame_stack, dataset.img_hw, dataset.img_hw),
            true_act_dim=dataset.act_dim,
            latent_act_dim=config.lapo.latent_action_dim,
            act_head_dim=config.lapo.act_head_dim,
            act_head_dropout=config.lapo.act_head_dropout,
            obs_head_dim=config.lapo.obs_head_dim,
            obs_head_dropout=config.lapo.obs_head_dropout,
            encoder_scale=config.lapo.encoder_scale,
            encoder_channels=(16, 32, 64, 128, 256) if config.lapo.encoder_deep else (16, 32, 32),
            encoder_num_res_blocks=config.lapo.encoder_num_res_blocks,
            encoder_dropout=config.lapo.encoder_dropout,
            encoder_norm_out=config.lapo.encoder_norm_out,
        ).to(DEVICE)
        load_checkpoint(lapo, None, None, config.lapo_checkpoint_path)
    else:
        print("=== Stage 1: LAPO Pretraining ===")
        lapo = train_mv(config=config.lapo, checkpoint_dir=checkpoint_dir, config_path=config_path)
    
    # Stage 2: latent behavioral cloning on frozen stage-1 labels.
    if config.bc_checkpoint_path:
        print("=== Stage 2: BC Pretraining (skipped, loading from checkpoint) ===")
        # Recreate the BC policy using dataset metadata before loading the checkpoint.
        dataset = DCSMVInMemoryDataset(
            config.bc.data_path, 
            max_offset=config.bc.fixed_offset, 
            frame_stack=config.bc.frame_stack, 
            device="cpu",
            camera_num=config.bc.camera_num,
            view_keys=config.bc.view_keys,
            resize_wh=(config.bc.resize_size, config.bc.resize_size),
            mixed_view_sampling=False,
            positive_samples_per_instance=4,
            action_key=config.bc.action_key,
            filter_noop=config.bc.filter_noop,
            num_trajectories_to_load=config.bc.num_trajectories_to_load,
        )
        actor = Actor(
            shape=(3 * config.bc.frame_stack, dataset.img_hw, dataset.img_hw),
            num_actions=config.lapo.latent_action_dim,
            encoder_scale=config.bc.encoder_scale,
            encoder_channels=(16, 32, 64, 128, 256) if config.bc.encoder_deep else (16, 32, 32),
            encoder_num_res_blocks=config.bc.encoder_num_res_blocks,
            dropout=config.bc.dropout,
        ).to(DEVICE)
        load_checkpoint(actor, None, None, config.bc_checkpoint_path)
    else:
        print("=== Stage 2: BC Pretraining ===")
        actor = train_bc(lam=lapo, config=config.bc, checkpoint_dir=checkpoint_dir, config_path=config_path)
    
    run.finish()
    return lapo

if __name__ == "__main__":
    train()
