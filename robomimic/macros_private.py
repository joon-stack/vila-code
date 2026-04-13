"""
Optional private overrides for robomimic macros.

This paper-release repo must not commit secrets. W&B is disabled by default.
If you want online logging, use `wandb login` or export your credentials
outside the repo instead of editing tracked files.
"""

DEBUG = False
VISUALIZE_RANDOMIZER = False
WANDB_ENTITY = None
WANDB_API_KEY = None
