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
    raise ModuleNotFoundError("Swanlab is required to log to SwanLab.") from None


class SwanlabSummaryWriter(SummaryWriter):
    """Summary writer for SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["swanlab_project"]
        except KeyError:
            raise KeyError("Please specify swanlab_project in the runner config, e.g. legged_gym.") from None

        # Initialize swanlab
        swanlab.init(
            project=project,
            experiment_name=run_name,
            logdir=log_dir,
        )

        # Add log directory to swanlab config
        swanlab.config.update({"log_dir": log_dir})

    def store_config(self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict) -> None:
        swanlab.config.update({"runner_cfg": runner_cfg})
        swanlab.config.update({"policy_cfg": policy_cfg})
        swanlab.config.update({"alg_cfg": alg_cfg})
        try:
            swanlab.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            swanlab.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        swanlab.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        swanlab.finish()

    def log_config(self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict) -> None:
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path: str, iter: int) -> None:
        # SwanLab uses watch to track files
        # Optionally you can log it as an artifact
        pass

    def save_file(self, path: str) -> None:
        # SwanLab uses watch to track files
        # Optionally you can log it as an artifact
        pass
