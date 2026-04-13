"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                if inputs["obs"][k].ndim - 2 != len(self.obs_shapes[k]):
                    assert inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k])
                else:
                    assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k]), "inputs[obs][k].shape: {}, self.obs_shapes[k]: {}".format(inputs["obs"][k].shape, self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]
            
            # # DEBUG: Check if ReLU activation is applied (causing negative values to be zeroed)
            # print(f"  - Number of negative values: {(obs_features < 0).sum().item()}")
            # print(f"  - Number of zero values: {(obs_features == 0).sum().item()}")
            # print(f"  - Number of positive values: {(obs_features > 0).sum().item()}")
            
            # # DEBUG: Check if encoder is in training mode
            # print(f"  - Encoder training mode: {self.nets['policy']['obs_encoder'].training}")
            # print(f"  - Policy training mode: {self.nets['policy'].training}")
            
            # # Force encoder to eval mode if it's still in training mode
            # if self.nets['policy']['obs_encoder'].training:
            #     print(f"  - FORCING encoder to eval mode!")
            #     self.nets['policy']['obs_encoder'].set_eval()
            
            # print("="*50 + "\n")

            obs_cond = obs_features.flatten(start_dim=1)
            
            # # DEBUG: Print diffusion policy condition information
            # print(f"\n=== DIFFUSION POLICY CONDITION DEBUG (Epoch: {epoch}) ===")
            # print(f"obs_features shape: {obs_features.shape}")
            # print(f"obs_features dtype: {obs_features.dtype}")
            # if torch.is_floating_point(obs_features):
            #     print(f"obs_features values (min, max, mean): {obs_features.min():.4f}, {obs_features.max():.4f}, {obs_features.mean():.4f}")
            #     print(f"obs_features std: {obs_features.std():.4f}")
            
            # print(f"obs_cond shape: {obs_cond.shape}")
            # print(f"obs_cond dtype: {obs_cond.dtype}")
            # if torch.is_floating_point(obs_cond):
            #     print(f"obs_cond values (min, max, mean): {obs_cond.min():.4f}, {obs_cond.max():.4f}, {obs_cond.mean():.4f}")
            #     print(f"obs_cond std: {obs_cond.std():.4f}")
            # print(f"=== END DIFFUSION POLICY CONDITION DEBUG ===\n")
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            # print(f"DEBUG: Passing global_cond to noise_pred_net:")
            # print(f"  - global_cond (obs_cond) shape: {obs_cond.shape}")
            # print(f"  - global_cond (obs_cond) dtype: {obs_cond.dtype}")
            # if torch.is_floating_point(obs_cond):
            #     print(f"  - global_cond values (min, max, mean): {obs_cond.min():.4f}, {obs_cond.max():.4f}, {obs_cond.mean():.4f}")
            
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            loss = F.mse_loss(noise_pred, noise)
            
            # logging
            losses = {
                "l2_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # pass
            # print("INFO: self.obs_shapes:", self.obs_shapes)
            # print("INFO: k:", k)
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension# if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k]), "inputs[obs][k].shape: {}, self.obs_shapes[k]: {}".format(inputs["obs"][k].shape, self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]
        # if torch.is_floating_point(obs_features):
        #     print(f"    Values (min, max, mean): {obs_features.min():.4f}, {obs_features.max():.4f}, {obs_features.mean():.4f}")
        # 
        # # DEBUG: Check if ReLU activation is applied (causing negative values to be zeroed)
        # print(f"  - Number of negative values: {(obs_features < 0).sum().item()}")
        # print(f"  - Number of zero values: {(obs_features == 0).sum().item()}")
        # print(f"  - Number of positive values: {(obs_features > 0).sum().item()}")
        # 
        # # DEBUG: Check if encoder is in training mode
        # print(f"  - Encoder training mode: {nets['policy']['obs_encoder'].training}")
        # print(f"  - Policy training mode: {nets['policy'].training}")
        # 
        # # Force encoder to eval mode if it's still in training mode
        # if nets['policy']['obs_encoder'].training:
        #     print(f"  - FORCING encoder to eval mode!")
        #     nets['policy']['obs_encoder'].set_eval()
        # 
        # print(f"=== END EVALUATION ENCODER DEBUG ===\n")

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)
        
        # # DEBUG: Print diffusion policy condition information during inference
        # print(f"\n=== DIFFUSION POLICY CONDITION DEBUG (INFERENCE) ===")
        # print(f"obs_features shape: {obs_features.shape}")
        # print(f"obs_features dtype: {obs_features.dtype}")
        # if torch.is_floating_point(obs_features):
        #     print(f"obs_features values (min, max, mean): {obs_features.min():.4f}, {obs_features.max():.4f}, {obs_features.mean():.4f}")
        #     print(f"obs_features std: {obs_features.std():.4f}")
        
        # print(f"obs_cond shape: {obs_cond.shape}")
        # print(f"obs_cond dtype: {obs_cond.dtype}")
        # if torch.is_floating_point(obs_cond):
        #     print(f"obs_cond values (min, max, mean): {obs_cond.min():.4f}, {obs_cond.max():.4f}, {obs_cond.mean():.4f}")
        #     print(f"obs_cond std: {obs_cond.std():.4f}")
        # print(f"=== END DIFFUSION POLICY CONDITION DEBUG (INFERENCE) ===\n")

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            # if k == self.noise_scheduler.timesteps[0]:  # Only print for first timestep to avoid spam
            #     print(f"DEBUG: Passing global_cond to noise_pred_net (inference):")
            #     print(f"  - global_cond (obs_cond) shape: {obs_cond.shape}")
            #     print(f"  - global_cond (obs_cond) dtype: {obs_cond.dtype}")
            #     if torch.is_floating_point(obs_cond):
            #         print(f"  - global_cond values (min, max, mean): {obs_cond.min():.4f}, {obs_cond.max():.4f}, {obs_cond.mean():.4f}")
            
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    @staticmethod
    def _remove_orig_mod_prefix(state_dict):
        """
        Remove _orig_mod prefix from state_dict keys if present.
        
        This handles checkpoints saved with torch.compile(), which wraps
        the original module with _orig_mod prefix.
        
        Args:
            state_dict (dict): state dictionary potentially containing _orig_mod prefix
            
        Returns:
            dict: state dictionary with _orig_mod prefix removed
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        # Remove _orig_mod prefix from state_dict keys if present (for torch.compile compatibility)
        print("INFO: self.nets: ", self.nets)
        nets_state_dict = self._remove_orig_mod_prefix(model_dict["nets"])
        print("INFO: nets_state_dict: ", nets_state_dict.keys())
        self.nets.load_state_dict(nets_state_dict)
        

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            ema_state_dict = self._remove_orig_mod_prefix(model_dict["ema"])
            self.ema.averaged_model.load_state_dict(ema_state_dict)

        if load_optimizers:
            try:
                for k in model_dict["optimizers"]:
                    self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
                for k in model_dict["lr_schedulers"]:
                    if model_dict["lr_schedulers"][k] is not None:
                        self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])
                print("INFO: Optimizer and LR scheduler states loaded successfully")
            except Exception as e:
                print(f"WARNING: Failed to load optimizer/scheduler state: {e}")
                print("WARNING: Optimizer and LR scheduler will be re-initialized")


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
