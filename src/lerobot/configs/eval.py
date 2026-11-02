# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime as dt
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from lerobot import envs, policies  # noqa: F401

from . import parser
from .default import EvalConfig, JobConfig
from .policies import PreTrainedConfig

logger = getLogger(__name__)

LEROBOT_GIT_URL = "https://github.com/huggingface/lerobot.git"


@dataclass
class EvalJobConfig(JobConfig):
    """`JobConfig` with evaluation defaults, for `lerobot-eval --job.target=<flavor>`.

    By default the pod runs the lerobot that ships in `image`. Set `lerobot_ref` to install lerobot
    from git on top of it instead, e.g. to evaluate with changes the image does not have yet.
    """

    # An evaluation is a bounded number of rollouts; a tighter cap than training's "2d" keeps a stuck
    # simulator from burning GPU time. Raise it for large protocols (e.g. all LIBERO suites).
    timeout: str | None = "6h"
    # Git ref (branch, tag or commit) of `lerobot_repo` to install on the pod. None uses the image's lerobot.
    lerobot_ref: str | None = None
    # Git repository `lerobot_ref` is taken from, e.g. your fork.
    lerobot_repo: str = LEROBOT_GIT_URL


@dataclass
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    # TF32 in float32 matmuls and cuDNN convolutions on Ampere and newer GPUs.
    tf32: bool | None = True
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    # Explicit consent to execute remote code from the Hub (required for hub environments).
    trust_remote_code: bool = False
    # Where the evaluation runs: omitted or "local" runs here, an HF Jobs flavor (e.g. "a10g-small")
    # submits it to HF Jobs. See `EvalJobConfig`.
    job: EvalJobConfig = field(default_factory=EvalJobConfig)
    # Upload eval_info.json, run metadata and a few videos to the Hub when the evaluation ends, under
    # `eval/<env.type>[-<env.task>]/<UTC timestamp>/`. None means on for HF Jobs runs (the job's disk
    # is discarded when it ends) and off for local runs.
    push_results_to_hub: bool | None = None
    # Model repo receiving the results. Defaults to the evaluated policy's Hub repo; set it to publish
    # elsewhere, e.g. when evaluating a policy you cannot write to. Created private if missing.
    results_repo_id: str | None = None
    # Number of rollout videos uploaded with the results (0 uploads none).
    results_max_videos: int = 1

    def __post_init__(self) -> None:
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=yaml_overrides + cli_overrides
            )
            self.policy.pretrained_path = Path(policy_path)

        else:
            logger.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type if self.policy is not None else 'scratch'}"
            else:
                self.job_name = (
                    f"{self.env.type}_{self.policy.type if self.policy is not None else 'scratch'}"
                )

            logger.warning(f"No job name provided, using '{self.job_name}' as job name.")

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

        if self.results_max_videos < 0:
            raise ValueError(f"results_max_videos must be >= 0, got {self.results_max_videos}.")

    @property
    def should_push_results(self) -> bool:
        """Whether this run uploads its results: `push_results_to_hub`, which defaults to on for HF Jobs."""
        if self.push_results_to_hub is None:
            return self.job.is_remote
        return self.push_results_to_hub

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
