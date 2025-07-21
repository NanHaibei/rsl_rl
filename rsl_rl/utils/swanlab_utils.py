# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import swanlab
except ModuleNotFoundError:
    raise ModuleNotFoundError("SwanLab is required to log to SwanLab platform. Please install with: pip install swanlab")


class SwanLabSummaryWriter(SummaryWriter):
    """Summary writer for SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["swanlab_project"]
        except KeyError:
            raise KeyError("Please specify swanlab_project in the runner config, e.g. legged_gym.")

        try:
            # SwanLab supports both SWANLAB_API_KEY and SWANLAB_WORKSPACE environment variables
            workspace = os.environ.get("SWANLAB_WORKSPACE", None)
        except KeyError:
            workspace = None

        # Initialize SwanLab
        swanlab.init(
            project=project, 
            workspace=workspace, 
            experiment_name=run_name,
            description=f"RL training experiment: {run_name}"
        )

        # Add log directory to SwanLab config
        swanlab.config.update({"log_dir": log_dir})

        # Name mapping for metric paths (similar to wandb)
        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        """Store configuration in SwanLab."""
        swanlab.config.update({"runner_cfg": runner_cfg})
        swanlab.config.update({"policy_cfg": policy_cfg})
        swanlab.config.update({"alg_cfg": alg_cfg})
        try:
            swanlab.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            swanlab.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        """Add scalar value to both TensorBoard and SwanLab."""
        # Log to TensorBoard
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        # Log to SwanLab
        swanlab.log({self._map_path(tag): scalar_value}, step=global_step)

    def stop(self):
        """Stop the SwanLab run."""
        swanlab.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        """Log configuration to SwanLab."""
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        """Save model artifact to SwanLab."""
        # SwanLab supports artifact uploading
        try:
            swanlab.save(model_path, base_path=os.path.dirname(model_path))
        except Exception as e:
            print(f"Warning: Failed to save model to SwanLab: {e}")

    def save_file(self, path, iter=None):
        """Save file artifact to SwanLab."""
        try:
            swanlab.save(path, base_path=os.path.dirname(path))
        except Exception as e:
            print(f"Warning: Failed to save file to SwanLab: {e}")

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        """Add video to both TensorBoard and SwanLab."""
        # Log to TensorBoard
        super().add_video(tag, vid_tensor, global_step=global_step, fps=fps, walltime=walltime)
        
        # SwanLab video logging (if supported)
        try:
            # Convert tensor to numpy if needed for SwanLab
            import numpy as np
            if hasattr(vid_tensor, 'numpy'):
                vid_numpy = vid_tensor.cpu().numpy()
            else:
                vid_numpy = vid_tensor
            
            # SwanLab expects video in specific format
            swanlab.log({
                f"{self._map_path(tag)}_video": swanlab.Video(vid_numpy, fps=fps)
            }, step=global_step)
        except Exception as e:
            print(f"Warning: Failed to log video to SwanLab: {e}")

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """Add image to both TensorBoard and SwanLab."""
        # Log to TensorBoard
        super().add_image(tag, img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)
        
        # SwanLab image logging
        try:
            import numpy as np
            if hasattr(img_tensor, 'numpy'):
                img_numpy = img_tensor.cpu().numpy()
            else:
                img_numpy = img_tensor
            
            swanlab.log({
                f"{self._map_path(tag)}_image": swanlab.Image(img_numpy)
            }, step=global_step)
        except Exception as e:
            print(f"Warning: Failed to log image to SwanLab: {e}")

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None):
        """Add histogram to both TensorBoard and SwanLab."""
        # Log to TensorBoard
        super().add_histogram(tag, values, global_step=global_step, bins=bins, walltime=walltime)
        
        # SwanLab histogram logging
        try:
            import numpy as np
            if hasattr(values, 'numpy'):
                values_numpy = values.cpu().numpy()
            else:
                values_numpy = values
            
            swanlab.log({
                f"{self._map_path(tag)}_hist": swanlab.Histogram(values_numpy)
            }, step=global_step)
        except Exception as e:
            print(f"Warning: Failed to log histogram to SwanLab: {e}")

    """
    Private methods.
    """

    def _map_path(self, path):
        """Map metric paths to SwanLab-friendly names."""
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
