import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


def set_seed(seed, env=None, deterministic_torch=False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def get_optim_groups(model, weight_decay):
    return [
        {"params": (p for p in model.parameters() if p.dim() < 2), "weight_decay": 0.0},
        {"params": (p for p in model.parameters() if p.dim() >= 2), "weight_decay": weight_decay},
    ]


def get_grad_norm(model):
    grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
    norm = torch.cat(grads).norm()
    return norm


def soft_update(target, source, tau=1e-3):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class DCSMVInMemoryDataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        frame_stack=1,
        device="cpu",
        max_offset=1,
        camera_num=8,
        view_keys=None,
        resize_wh=(128, 128),
        view_keys_to_load=None,
        num_trajectories_to_load=None,
        selected_demo_names=None,
        mixed_view_sampling=False,
        positive_samples_per_instance=4,
        action_key="actions",
        filter_noop=False,
        real_sort=False,
        use_depth=False,
        depth_key_suffix="agentview_depth",
    ):
        self.frame_stack = frame_stack
        self.device = device
        self.max_offset = max_offset
        self.camera_num = camera_num
        self.view_keys = view_keys or []
        self.resize_wh = resize_wh
        self.mixed_view_sampling = mixed_view_sampling
        self.positive_samples_per_instance = positive_samples_per_instance
        self.action_key = action_key
        self.real_sort = real_sort
        self.use_depth = use_depth
        self.depth_key_suffix = depth_key_suffix
        self.filter_noop = filter_noop

        self.view_keys_to_load = view_keys_to_load if view_keys_to_load is not None else self.view_keys
        self.selected_demo_names = set(selected_demo_names) if selected_demo_names is not None else None

        self.data_dict = {view: [] for view in self.view_keys_to_load}
        if self.use_depth:
            self.depth_dict = {view: [] for view in self.view_keys_to_load}
        self.load_multi_view_data(hdf5_path, num_trajectories_to_load)

        self.actions = []
        self.states = []
        self.load_actions_states(hdf5_path, num_trajectories_to_load, action_key)

        if self.view_keys_to_load and not self.data_dict[self.view_keys_to_load[0]]:
            raise ValueError("No trajectories loaded. Check your HDF5 file path and view keys.")
        self.traj_lens = [len(traj) for traj in self.data_dict[self.view_keys_to_load[0]]]

        min_traj_len = min(self.traj_lens) if self.traj_lens else 0
        if min_traj_len <= max_offset:
            print(
                f"Warning: max_offset ({max_offset}) is not smaller than the minimum trajectory length "
                f"({min_traj_len}). Adjusting max_offset."
            )
            self.max_offset = min_traj_len - 1
        else:
            self.max_offset = max_offset

        self.img_hw = self.resize_wh[0]
        self.total_moments = 0
        for traj_len in self.traj_lens:
            self.total_moments += max(0, traj_len - self.max_offset)

        self.moment_to_traj_transition = []
        for traj_idx, traj_len in enumerate(self.traj_lens):
            valid_transitions = max(0, traj_len - self.max_offset)
            for transition_idx in range(valid_transitions):
                self.moment_to_traj_transition.append((traj_idx, transition_idx))

        self.fixed_offset = None

    def set_fixed_offset(self, offset):
        self.fixed_offset = offset

    def load_multi_view_data(self, hdf5_path, num_trajectories_to_load=None):
        print(f"Loading multi-view data from HDF5 file: {hdf5_path}...")
        if self.use_depth:
            print(f"Depth loading enabled with suffix: {self.depth_key_suffix}")
        try:
            with h5py.File(hdf5_path, "r") as f:
                demos = sorted(list(f["data"].keys()))
                if self.selected_demo_names is not None:
                    demos = [d for d in demos if d in self.selected_demo_names]
                if num_trajectories_to_load is not None and not self.real_sort:
                    demos = demos[:num_trajectories_to_load]

                self.demo_names = list(demos)

                for demo in tqdm(demos, desc="Loading Multi-view Demos"):
                    temp_traj_data = {view: [] for view in self.view_keys_to_load}
                    temp_depth_data = {view: [] for view in self.view_keys_to_load} if self.use_depth else None

                    is_demo_valid = True
                    for view_key in self.view_keys_to_load:
                        full_key = f"data/{demo}/obs/{view_key}"
                        if full_key not in f:
                            print(f"Warning: View {view_key} not found in demo {demo}. Skipping this demo.")
                            is_demo_valid = False
                            break
                        if self.use_depth:
                            depth_key = view_key.replace("agentview_image", self.depth_key_suffix)
                            depth_full_key = f"data/{demo}/obs/{depth_key}"
                            if depth_full_key not in f:
                                print(f"Warning: Depth {depth_key} not found in demo {demo}. Skipping this demo.")
                                is_demo_valid = False
                                break
                    if not is_demo_valid:
                        continue

                    for view_key in self.view_keys_to_load:
                        images = f[f"data/{demo}/obs/{view_key}"][()]
                        resized_images = [np.array(Image.fromarray(img).resize(self.resize_wh)) for img in images]
                        temp_traj_data[view_key] = np.stack(resized_images)

                        if self.use_depth:
                            depth_key = view_key.replace("agentview_image", self.depth_key_suffix)
                            depth_images = f[f"data/{demo}/obs/{depth_key}"][()]
                            import cv2

                            resized_depth = []
                            for d_img in depth_images:
                                d_squeezed = d_img.squeeze(-1) if d_img.ndim == 3 else d_img
                                d_resized = cv2.resize(d_squeezed, self.resize_wh, interpolation=cv2.INTER_LINEAR)
                                resized_depth.append(d_resized[..., np.newaxis])
                            temp_depth_data[view_key] = np.stack(resized_depth)

                    for view_key in self.view_keys_to_load:
                        self.data_dict[view_key].append(temp_traj_data[view_key])
                        if self.use_depth:
                            self.depth_dict[view_key].append(temp_depth_data[view_key])

        except Exception as e:
            print(f"Error loading HDF5 file: {e}")
            raise

        print("Multi-view data loaded successfully.")
        if self.use_depth:
            print("Depth data loaded successfully.")

    def load_actions_states(self, hdf5_path, num_trajectories_to_load=None, action_key="actions"):
        print("Loading actions and states...")
        try:
            with h5py.File(hdf5_path, "r") as f:
                demos = sorted(list(f["data"].keys()))
                if self.selected_demo_names is not None:
                    demos = [d for d in demos if d in self.selected_demo_names]
                if num_trajectories_to_load is not None:
                    demos = demos[:num_trajectories_to_load]

                if hasattr(self, "demo_names"):
                    demos = [d for d in self.demo_names if d in set(demos)]

                for demo in tqdm(demos, desc="Loading Actions and States"):
                    is_demo_valid = all(f"data/{demo}/obs/{view_key}" in f for view_key in self.view_keys_to_load)
                    if not is_demo_valid:
                        continue

                    if f"data/{demo}/{action_key}" in f:
                        actions = f[f"data/{demo}/{action_key}"][()]
                        self.actions.append(actions)

                    if f"data/{demo}/states" in f:
                        states = f[f"data/{demo}/states"][()]
                        self.states.append(states)

                if self.actions:
                    self.act_dim = self.actions[0].shape[-1]
                if self.states:
                    self.state_dim = self.states[0].shape[-1]

                if "action_mean" in f.attrs:
                    self.action_mean = f.attrs["action_mean"]
                    print("INFO: self.action_mean", self.action_mean)
                else:
                    self.action_mean = None
                if "action_std" in f.attrs:
                    self.action_std = f.attrs["action_std"]
                    print("INFO: self.action_std", self.action_std)
                else:
                    self.action_std = None

        except Exception as e:
            print(f"Error loading actions/states: {e}")
            raise

    def _get_single_view_padded_obs(self, traj_idx, idx, view_key):
        traj_view_obs = self.data_dict[view_key][traj_idx]
        traj_len = len(traj_view_obs)

        if traj_len == 0:
            channels_per_frame = 4 if self.use_depth else 3
            dummy_shape = (*self.resize_wh, self.frame_stack * channels_per_frame)
            return torch.zeros(dummy_shape, device=self.device, dtype=torch.float32 if self.use_depth else torch.uint8)

        idx = min(idx, traj_len - 1)

        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = min(idx + 1, traj_len)
        obs_rgb = traj_view_obs[min_obs_idx:max_obs_idx]

        if len(obs_rgb) < self.frame_stack:
            pad_img = obs_rgb[0][None]
            obs_rgb = np.concatenate([pad_img] * (self.frame_stack - len(obs_rgb)) + [obs_rgb])

        obs_rgb = torch.as_tensor(np.array(obs_rgb), device=self.device)
        obs_rgb = obs_rgb.permute((1, 2, 0, 3))
        obs_rgb = obs_rgb.reshape(*obs_rgb.shape[:2], -1)

        if self.use_depth:
            traj_depth_obs = self.depth_dict[view_key][traj_idx]
            obs_depth = traj_depth_obs[min_obs_idx:max_obs_idx]

            if len(obs_depth) < self.frame_stack:
                pad_depth = obs_depth[0][None]
                obs_depth = np.concatenate([pad_depth] * (self.frame_stack - len(obs_depth)) + [obs_depth])

            obs_depth = torch.as_tensor(np.array(obs_depth), device=self.device, dtype=torch.float32)
            obs_depth = obs_depth.permute((1, 2, 0, 3))
            obs_depth = obs_depth.reshape(*obs_depth.shape[:2], -1)
            obs_rgb = obs_rgb.float()
            return torch.cat([obs_rgb, obs_depth], dim=-1)

        return obs_rgb

    def __len__(self):
        return self.total_moments

    def __getitem__(self, idx):
        if idx >= len(self.moment_to_traj_transition):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self.moment_to_traj_transition)} moments"
            )

        traj_idx, transition_idx = self.moment_to_traj_transition[idx]
        all_views_obs = []
        all_views_next_obs = []
        all_views_future_obs = []
        obs_camera_indices = []
        future_camera_indices = []

        current_traj_len = self.traj_lens[traj_idx]
        max_possible_offset = current_traj_len - transition_idx - 1
        if self.fixed_offset is not None:
            offset = min(self.fixed_offset, max_possible_offset)
        else:
            offset = random.randint(1, min(self.max_offset, max_possible_offset))

        if self.mixed_view_sampling:
            for _ in range(self.positive_samples_per_instance):
                obs_view = random.choice(self.view_keys_to_load)
                next_obs_view = random.choice(self.view_keys_to_load)
                future_obs_view = random.choice(self.view_keys_to_load)

                obs = self._get_single_view_padded_obs(traj_idx, transition_idx, obs_view)
                next_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + 1, next_obs_view)
                future_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, future_obs_view)

                all_views_obs.append(obs)
                all_views_next_obs.append(next_obs)
                all_views_future_obs.append(future_obs)
                obs_camera_indices.append(self.view_keys_to_load.index(obs_view))
                future_camera_indices.append(self.view_keys_to_load.index(future_obs_view))
        else:
            for view_key in self.view_keys_to_load:
                obs = self._get_single_view_padded_obs(traj_idx, transition_idx, view_key)
                next_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + 1, view_key)
                future_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, view_key)

                all_views_obs.append(obs)
                all_views_next_obs.append(next_obs)
                all_views_future_obs.append(future_obs)

                view_idx = self.view_keys_to_load.index(view_key)
                obs_camera_indices.append(view_idx)
                future_camera_indices.append(view_idx)

        action = torch.tensor(self.actions[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
        state = torch.tensor(self.states[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
        action_sequence = self.actions[traj_idx][transition_idx : transition_idx + offset]
        action_sequence = torch.tensor(action_sequence, device=self.device, dtype=torch.float32)
        instance_id = traj_idx * 10000 + transition_idx

        return {
            "obs": torch.stack(all_views_obs),
            "next_obs": torch.stack(all_views_next_obs),
            "future_obs": torch.stack(all_views_future_obs),
            "action": action,
            "action_sequence": action_sequence,
            "state": state,
            "instance_id": instance_id,
            "offset": offset - 1,
            "obs_camera_idx": torch.tensor(obs_camera_indices, dtype=torch.long),
            "future_camera_idx": torch.tensor(future_camera_indices, dtype=torch.long),
        }


def normalize_img(img, use_depth=False, num_depth_channels=0):
    if use_depth and num_depth_channels > 0:
        rgb_channels = img.shape[1] - num_depth_channels
        rgb = img[:, :rgb_channels]
        depth = img[:, rgb_channels:]
        rgb_norm = ((rgb / 255.0) - 0.5) * 2.0
        depth_norm = depth
        return torch.cat([rgb_norm, depth_norm], dim=1)
    return ((img / 255.0) - 0.5) * 2.0


def metric_learning_collate_fn(batch_list, K=4, mixed_view_sampling=False):
    P = len(batch_list)

    final_obs = []
    final_next_obs = []
    final_future_obs = []
    final_actions = []
    unpadded_action_sequences = []
    final_states = []
    final_offsets = []
    final_instance_ids = []
    for i in range(P):
        instance_group = batch_list[i]

        if mixed_view_sampling:
            num_positive_samples = instance_group["obs"].shape[0]

            final_obs.append(instance_group["obs"])
            final_next_obs.append(instance_group["next_obs"])
            final_future_obs.append(instance_group["future_obs"])
            final_actions.append(instance_group["action"].unsqueeze(0).repeat(num_positive_samples, 1))
            for _ in range(num_positive_samples):
                unpadded_action_sequences.append(instance_group["action_sequence"])
            final_states.append(instance_group["state"].unsqueeze(0).repeat(num_positive_samples, 1))
            final_offsets.append(
                torch.tensor([instance_group["offset"]] * num_positive_samples, device=instance_group["action"].device)
            )
            final_instance_ids.append(torch.tensor([instance_group["instance_id"]] * num_positive_samples))
        else:
            num_available_views = instance_group["obs"].shape[0]
            k_sample = K
            if k_sample == 0:
                continue

            if K > num_available_views:
                sampled_indices = random.choices(range(num_available_views), k=k_sample)
            else:
                sampled_indices = random.sample(range(num_available_views), k_sample)

            final_obs.append(instance_group["obs"][sampled_indices])
            final_next_obs.append(instance_group["next_obs"][sampled_indices])
            final_future_obs.append(instance_group["future_obs"][sampled_indices])
            final_actions.append(instance_group["action"].unsqueeze(0).repeat(k_sample, 1))
            for _ in range(k_sample):
                unpadded_action_sequences.append(instance_group["action_sequence"])
            final_states.append(instance_group["state"].unsqueeze(0).repeat(k_sample, 1))
            final_offsets.append(torch.tensor([instance_group["offset"]] * k_sample, device=instance_group["action"].device))
            final_instance_ids.append(torch.tensor([instance_group["instance_id"]] * k_sample))

    if not final_obs:
        return None

    padded_action_sequences = pad_sequence(unpadded_action_sequences, batch_first=True, padding_value=0.0)
    action_sequences_mask = torch.zeros(padded_action_sequences.shape[:2], device=padded_action_sequences.device)
    for i, seq in enumerate(unpadded_action_sequences):
        action_sequences_mask[i, : len(seq)] = 1.0

    flat_padded_action_sequences = padded_action_sequences.flatten(start_dim=1)

    return {
        "obs": torch.cat(final_obs, dim=0),
        "next_obs": torch.cat(final_next_obs, dim=0),
        "future_obs": torch.cat(final_future_obs, dim=0),
        "actions": torch.cat(final_actions, dim=0),
        "action_sequences": flat_padded_action_sequences,
        "action_sequences_mask": action_sequences_mask,
        "states": torch.cat(final_states, dim=0),
        "offsets": torch.cat(final_offsets, dim=0),
        "instance_ids": torch.cat(final_instance_ids, dim=0),
    }


def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
