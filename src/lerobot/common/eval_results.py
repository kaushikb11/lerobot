# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Publish `lerobot-eval` results to a Hub model repo.

Each run lands in its own folder, `eval/<env.type>[-<env.task>]/<UTC timestamp>/`, holding:

- `eval_info.json`: the file `lerobot-eval` writes locally, uploaded unchanged, so tools that read it
  (e.g. `lerobot-eval-compare`) work on a downloaded copy as-is.
- `eval_metadata.json`: provenance needed to reproduce or compare the run (policy repo and the commit
  that was evaluated, lerobot version, seed, episodes, environment) plus a copy of the overall success
  summary, and the full eval config.
- `videos/`: the first few rollout videos, if any were rendered.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import draccus
from huggingface_hub import CommitOperationAdd, HfApi

from lerobot.__version__ import __version__

if TYPE_CHECKING:
    from lerobot.configs.eval import EvalPipelineConfig

logger = logging.getLogger(__name__)

EVAL_RESULTS_ROOT = "eval"
EVAL_INFO_FILENAME = "eval_info.json"
EVAL_METADATA_FILENAME = "eval_metadata.json"
# Bumped when a key of eval_metadata.json changes meaning or is removed; adding keys keeps the version.
EVAL_METADATA_FORMAT_VERSION = 1
# Logged once the results are on the Hub; `lerobot.jobs.eval` watches for it to finish following the
# job early. Keep the two in sync (a mismatch only makes the client wait for the job stage instead).
RESULTS_PUSHED_MARKER = "Eval results pushed to https://huggingface.co/"

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_MAX_SLUG_LEN = 96
# The `overall` keys copied into the metadata summary (everything but the per-file video lists).
_SUMMARY_KEYS = (
    "pc_success",
    "n_success",
    "n_episodes",
    "pc_success_ci95",
    "avg_sum_reward",
    "avg_max_reward",
)


def policy_hub_repo_id(cfg: EvalPipelineConfig) -> str | None:
    """The Hub repo id the evaluated policy was loaded from, or None for a local or from-scratch policy."""
    if cfg.policy is None or cfg.policy.pretrained_path is None:
        return None
    path = str(cfg.policy.pretrained_path)
    return None if os.path.exists(path) else path


def resolve_results_repo_id(cfg: EvalPipelineConfig) -> str:
    """The model repo the results go to: `--results_repo_id`, else the evaluated policy's own repo.

    Raises:
        ValueError: When neither is available (a local or from-scratch policy without
            `--results_repo_id`).
    """
    if cfg.results_repo_id:
        return cfg.results_repo_id
    repo_id = policy_hub_repo_id(cfg)
    if repo_id is None:
        raise ValueError(
            "--push_results_to_hub needs a Hub repo to publish to: evaluate a policy from the Hub "
            "(--policy.path=<user>/<repo>) or pass --results_repo_id=<user>/<repo>."
        )
    return repo_id


def resolve_policy_revision(cfg: EvalPipelineConfig, api: HfApi | None = None) -> str | None:
    """Commit sha of the evaluated Hub policy, so the published results name exactly what was run.

    Resolve it before evaluating: publishing the results adds a commit to the policy repo, so the
    repo's head afterwards is no longer the evaluated revision. Returns None for a local policy.
    """
    repo_id = policy_hub_repo_id(cfg)
    if repo_id is None or cfg.policy is None:
        return None
    api = api or HfApi()
    return api.model_info(repo_id, revision=cfg.policy.pretrained_revision).sha


def results_slug(cfg: EvalPipelineConfig) -> str:
    """The `<env.type>[-<env.task>]` folder name, filesystem- and URL-safe."""
    parts = [cfg.env.type]
    task = getattr(cfg.env, "task", None)
    if task:
        parts.append(str(task))
    slug = _SLUG_RE.sub("-", "-".join(parts)).strip("-")
    return slug[:_MAX_SLUG_LEN] or "env"


def results_path_in_repo(cfg: EvalPipelineConfig, now: dt.datetime) -> str:
    """Folder of one run inside the results repo: `eval/<slug>/<YYYY-MM-DD_HH-MM-SS>` (UTC)."""
    return f"{EVAL_RESULTS_ROOT}/{results_slug(cfg)}/{now.astimezone(dt.UTC):%Y-%m-%d_%H-%M-%S}"


def _json_safe(value: Any) -> Any:
    """Replace NaN/inf (e.g. `pc_success` of an empty group) with None: they are not valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def build_eval_metadata(
    cfg: EvalPipelineConfig, info: dict[str, Any], policy_revision: str | None, now: dt.datetime
) -> dict[str, Any]:
    """Provenance and summary written next to `eval_info.json`."""
    overall = info.get("overall", {})
    config = draccus.encode(cfg)
    if isinstance(config, dict):
        # Client-only orchestration; the run itself always executes locally to its machine.
        config.pop("job", None)
    return _json_safe(
        {
            "format_version": EVAL_METADATA_FORMAT_VERSION,
            "created_at": now.astimezone(dt.UTC).isoformat(timespec="seconds"),
            "lerobot_version": __version__,
            "policy": {
                "repo_id": policy_hub_repo_id(cfg),
                "revision": policy_revision,
                "type": cfg.policy.type if cfg.policy is not None else None,
            },
            "env": {"type": cfg.env.type, "task": getattr(cfg.env, "task", None)},
            "seed": cfg.seed,
            # Requested episodes per task; the evaluated count is in `summary.n_episodes`.
            "n_episodes_per_task": cfg.eval.n_episodes,
            "summary": {key: overall.get(key) for key in _SUMMARY_KEYS},
            "config": config,
        }
    )


def push_eval_results(
    cfg: EvalPipelineConfig,
    info: dict[str, Any],
    *,
    policy_revision: str | None,
    api: HfApi | None = None,
    now: dt.datetime | None = None,
) -> str:
    """Upload one evaluation run to the results repo in a single commit.

    The results repo defaults to the evaluated policy's repo. A different `--results_repo_id` is
    created (private) when it does not exist yet.

    Args:
        cfg: The eval config of the run.
        info: The eval info dict, as written to `eval_info.json`.
        policy_revision: The evaluated policy commit, from `resolve_policy_revision`.
        api: Hub client to use (a default `HfApi()` otherwise).
        now: Timestamp of the run folder (current UTC time otherwise).

    Returns:
        The URL of the uploaded run folder.
    """
    api = api or HfApi()
    now = now or dt.datetime.now(dt.UTC)
    repo_id = resolve_results_repo_id(cfg)
    if repo_id != policy_hub_repo_id(cfg):
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    folder = results_path_in_repo(cfg, now)
    metadata = build_eval_metadata(cfg, info, policy_revision, now)
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{folder}/{EVAL_INFO_FILENAME}",
            path_or_fileobj=json.dumps(info, indent=2).encode(),
        ),
        CommitOperationAdd(
            path_in_repo=f"{folder}/{EVAL_METADATA_FILENAME}",
            path_or_fileobj=json.dumps(metadata, indent=2).encode(),
        ),
    ]
    videos_root = Path(cfg.output_dir) / "videos" if cfg.output_dir is not None else None
    for video in [Path(p) for p in info.get("overall", {}).get("video_paths", [])][: cfg.results_max_videos]:
        if not video.is_file():
            continue
        try:
            rel = video.relative_to(videos_root) if videos_root is not None else Path(video.name)
        except ValueError:
            rel = Path(video.name)
        operations.append(
            CommitOperationAdd(path_in_repo=f"{folder}/videos/{rel.as_posix()}", path_or_fileobj=video)
        )

    summary = metadata["summary"]
    pc_success = summary.get("pc_success")
    headline = f"{pc_success:.1f}% success" if isinstance(pc_success, float) else "no success rate"
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=(
            f"Add eval results for {results_slug(cfg)}: {headline} "
            f"({summary.get('n_success')}/{summary.get('n_episodes')} episodes)"
        ),
    )
    url = f"https://huggingface.co/{repo_id}/tree/main/{folder}"
    logger.info("%s%s/tree/main/%s", RESULTS_PUSHED_MARKER, repo_id, folder)
    return url
