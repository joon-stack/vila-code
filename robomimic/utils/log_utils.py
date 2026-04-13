"""
This file contains utility classes and functions for logging to stdout, stderr,
and to tensorboard.
"""
import os
import sys
import numpy as np
from datetime import datetime
from contextlib import contextmanager
import textwrap
import time
from tqdm import tqdm
from termcolor import colored

import robomimic

# global list of warning messages can be populated with @log_warning and flushed with @flush_warnings
WARNINGS_BUFFER = []


class PrintLogger(object):
    """
    This class redirects print statements to both console and a file.
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # ensure stdout gets flushed
        self.terminal.flush()

    def isatty(self):
        return False


class DataLogger(object):
    """
    Logging class to log metrics to tensorboard and/or retrieve running statistics about logged data.
    """
    def __init__(self, log_dir, config, log_tb=True, log_wandb=False):
        """
        Args:
            log_dir (str): base path to store logs
            log_tb (bool): whether to use tensorboard logging
        """
        self._tb_logger = None
        self._wandb_logger = None
        self._data = dict() # store all the scalar data logged so far

        if log_tb:
            from tensorboardX import SummaryWriter
            self._tb_logger = SummaryWriter(os.path.join(log_dir, 'tb'))

        if log_wandb:
            import wandb
            import robomimic.macros as Macros
            
            # set up wandb api key if specified in macros
            if Macros.WANDB_API_KEY is not None:
                os.environ["WANDB_API_KEY"] = Macros.WANDB_API_KEY

            assert Macros.WANDB_ENTITY is not None, "WANDB_ENTITY macro is set to None." \
                    "\nSet this macro in {base_path}/macros_private.py" \
                    "\nIf this file does not exist, first run python {base_path}/scripts/setup_macros.py".format(base_path=robomimic.__path__[0])
            
            # attempt to set up wandb 10 times. If unsuccessful after these trials, don't use wandb
            num_attempts = 10
            for attempt in range(num_attempts):
                try:
                    # set up wandb
                    self._wandb_logger = wandb

                    # build helpful tags for better visual grouping in W&B
                    tags = []
                    try:
                        tags.append(str(config.algo_name))
                    except Exception:
                        pass
                    try:
                        seed = config.train.seed
                        tags.append(f"seed:{seed}")
                    except Exception:
                        pass
                    try:
                        # number of rgb views (if present)
                        num_views = len(config.observation.modalities.obs.rgb)
                        tags.append(f"views:{num_views}")
                    except Exception:
                        pass
                    try:
                        fd = config.observation.encoder.rgb.core_kwargs.get('feature_dimension', None)
                        if fd is None:
                            tags.append("fd:none")
                        else:
                            tags.append(f"fd:{fd}")
                    except Exception:
                        pass

                    self._wandb_logger.init(
                        entity=Macros.WANDB_ENTITY,
                        project=config.experiment.logging.wandb_proj_name,
                        name=config.experiment.name,
                        dir=log_dir,
                        mode=("offline" if attempt == num_attempts - 1 else "online"),
                        tags=tags,
                    )

                    # set up info for identifying experiment
                    wandb_config = {k: v for (k, v) in config.meta.items() if k not in ["hp_keys", "hp_values"]}
                    for (k, v) in zip(config.meta["hp_keys"], config.meta["hp_values"]):
                        wandb_config[k] = v
                    if "algo" not in wandb_config:
                        wandb_config["algo"] = config.algo_name
                    self._wandb_logger.config.update(wandb_config)

                    # set a human-readable run name (post-init is allowed)
                    try:
                        # Only use the name from the config file, do not append tags.
                        run_name = str(config.experiment.name)
                        self._wandb_logger.run.name = run_name
                        self._wandb_logger.run.save()
                    except Exception:
                        pass

                    # Also push the full resolved config so WandB shows it clearly
                    try:
                        import json as _json
                        full_cfg = config.dump()
                        # ensure json-serializable (remove numpy types, etc.)
                        full_cfg = _json.loads(_json.dumps(full_cfg))
                        # 1) expose at top-level for filtering;
                        self._wandb_logger.config.update(full_cfg, allow_val_change=True)
                        # 2) keep a nested copy for convenience
                        self._wandb_logger.config.update({"config_full": full_cfg}, allow_val_change=True)
                    except Exception as _:
                        pass

                    # Provide a concise summary for quick scanning
                    try:
                        brief = {
                            "algo": config.algo_name,
                            "seed": getattr(config.train, 'seed', None),
                            "epochs": getattr(config.train, 'num_epochs', None),
                            "batch_size": getattr(config.train, 'batch_size', None),
                            "obs_horizon": getattr(config.algo.horizon, 'observation_horizon', None) if hasattr(config.algo, 'horizon') else None,
                            "act_horizon": getattr(config.algo.horizon, 'action_horizon', None) if hasattr(config.algo, 'horizon') else None,
                            "pred_horizon": getattr(config.algo.horizon, 'prediction_horizon', None) if hasattr(config.algo, 'horizon') else None,
                        }
                        try:
                            brief["config_file"] = os.path.basename(config.meta.config_path)
                        except Exception:
                            pass
                        try:
                            # For multiview experiments, try to get original RGB views from config file
                            try:
                                if hasattr(config.meta, 'config_path'):
                                    import json
                                    with open(config.meta.config_path, 'r') as f:
                                        original_config = json.load(f)
                                    brief["rgb_views"] = original_config.get("observation", {}).get("modalities", {}).get("obs", {}).get("rgb", [])
                                else:
                                    brief["rgb_views"] = config.observation.modalities.obs.rgb
                            except Exception:
                                brief["rgb_views"] = config.observation.modalities.obs.rgb
                        except Exception:
                            pass
                        try:
                            encoder_cfg = config.observation.encoder.rgb
                            brief["encoder_core"] = encoder_cfg.core_class
                            brief["encoder_backbone"] = encoder_cfg.core_kwargs.backbone_class
                            brief["encoder_feat_dim"] = encoder_cfg.core_kwargs.get('feature_dimension', None)
                            if "backbone_kwargs" in encoder_cfg.core_kwargs and "model_path" in encoder_cfg.core_kwargs.backbone_kwargs:
                                # Store only filename and parent directory
                                full_path = encoder_cfg.core_kwargs.backbone_kwargs.model_path
                                path_parts = full_path.split('/')
                                if len(path_parts) >= 2:
                                    brief["encoder_pretrain_path"] = f"{path_parts[-2]}/{path_parts[-1]}"
                                else:
                                    brief["encoder_pretrain_path"] = os.path.basename(full_path)
                        except Exception:
                            pass
                        try:
                            # full dataset paths and trajectory splits
                            d = config.train.data
                            if isinstance(d, list):
                                brief["datasets"] = [item['path'] for item in d if isinstance(item, dict) and 'path' in item]
                                traj_splits = [item['traj_split_path'] for item in d if isinstance(item, dict) and 'traj_split_path' in item]
                                if traj_splits:
                                    # Store only filename and parent directory for trajectory splits
                                    brief_traj_splits = []
                                    for split_path in traj_splits:
                                        path_parts = split_path.split('/')
                                        if len(path_parts) >= 2:
                                            brief_traj_splits.append(f"{path_parts[-2]}/{path_parts[-1]}")
                                        else:
                                            brief_traj_splits.append(os.path.basename(split_path))
                                    brief["traj_split_paths"] = brief_traj_splits
                            elif isinstance(d, str):
                                brief["datasets"] = [d]
                        except Exception:
                            pass
                        self._wandb_logger.config.update({"config_brief": brief}, allow_val_change=True)
                    except Exception:
                        pass

                    break
                except Exception as e:
                    log_warning("wandb initialization error (attempt #{}): {}".format(attempt + 1, e))
                    self._wandb_logger = None
                    time.sleep(30)

    def record(self, k, v, epoch, data_type='scalar', log_stats=False):
        """
        Record data with logger.
        Args:
            k (str): key string
            v (float or image): value to store
            epoch: current epoch number
            data_type (str): the type of data. either 'scalar' or 'image'
            log_stats (bool): whether to store the mean/max/min/std for all data logged so far with key k
        """

        assert data_type in ['scalar', 'image']

        if data_type == 'scalar':
            # maybe update internal cache if logging stats for this key
            if log_stats or k in self._data: # any key that we're logging or previously logged
                if k not in self._data:
                    self._data[k] = []
                self._data[k].append(v)

        # maybe log to tensorboard
        if self._tb_logger is not None:
            if data_type == 'scalar':
                self._tb_logger.add_scalar(k, v, epoch)
                if log_stats:
                    stats = self.get_stats(k)
                    for (stat_k, stat_v) in stats.items():
                        stat_k_name = '{}-{}'.format(k, stat_k)
                        self._tb_logger.add_scalar(stat_k_name, stat_v, epoch)
            elif data_type == 'image':
                if len(v.shape) == 3:
                    v = v[None, ...]
                self._tb_logger.add_images(k, img_tensor=v, global_step=epoch, dataformats="NHWC")

        if self._wandb_logger is not None:
            try:
                if data_type == 'scalar':
                    self._wandb_logger.log({k: v}, step=epoch)
                    if log_stats:
                        stats = self.get_stats(k)
                        for (stat_k, stat_v) in stats.items():
                            self._wandb_logger.log({"{}/{}".format(k, stat_k): stat_v}, step=epoch)
                elif data_type == 'image':
                    import wandb
                    self._wandb_logger.log({k: wandb.Image(v)}, step=epoch)
            except Exception as e:
                log_warning("wandb logging: {}".format(e))

    def get_stats(self, k):
        """
        Computes running statistics for a particular key.
        Args:
            k (str): key string
        Returns:
            stats (dict): dictionary of statistics
        """
        stats = dict()
        stats['mean'] = np.mean(self._data[k])
        stats['std'] = np.std(self._data[k])
        stats['min'] = np.min(self._data[k])
        stats['max'] = np.max(self._data[k])
        return stats

    def close(self):
        """
        Run before terminating to make sure all logs are flushed
        """
        if self._tb_logger is not None:
            self._tb_logger.close()

        if self._wandb_logger is not None:
            self._wandb_logger.finish()


class custom_tqdm(tqdm):
    """
    Small extension to tqdm to make a few changes from default behavior.
    By default tqdm writes to stderr. Instead, we change it to write
    to stdout.
    """
    def __init__(self, *args, **kwargs):
        assert "file" not in kwargs
        super(custom_tqdm, self).__init__(*args, file=sys.stdout, **kwargs)


@contextmanager
def silence_stdout():
    """
    This contextmanager will redirect stdout so that nothing is printed
    to the terminal. Taken from the link below:

    https://stackoverflow.com/questions/6735917/redirecting-stdout-to-nothing-in-python
    """
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target


def log_warning(message, color="yellow", print_now=True):
    """
    This function logs a warning message by recording it in a global warning buffer.
    The global registry will be maintained until @flush_warnings is called, at
    which point the warnings will get printed to the terminal.

    Args:
        message (str): warning message to display
        color (str): color of message - defaults to "yellow"
        print_now (bool): if True (default), will print to terminal immediately, in
            addition to adding it to the global warning buffer
    """
    global WARNINGS_BUFFER
    buffer_message = colored("ROBOMIMIC WARNING(\n{}\n)".format(textwrap.indent(message, "    ")), color)
    WARNINGS_BUFFER.append(buffer_message)
    if print_now:
        print(buffer_message)


def flush_warnings():
    """
    This function flushes all warnings from the global warning buffer to the terminal and
    clears the global registry.
    """
    global WARNINGS_BUFFER
    for msg in WARNINGS_BUFFER:
        print(msg)
    WARNINGS_BUFFER = []
