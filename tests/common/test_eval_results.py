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

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import draccus
import pytest

from lerobot.__version__ import __version__
from lerobot.common.eval_results import (
    RESULTS_PUSHED_MARKER,
    build_eval_metadata,
    push_eval_results,
    resolve_policy_revision,
    resolve_results_repo_id,
    results_path_in_repo,
    results_slug,
)
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401  (registers --policy.type act)

NOW = dt.datetime(2026, 11, 2, 9, 30, 5, tzinfo=dt.UTC)


def _cfg(monkeypatch, *args, policy_path=None):
    """Parse an EvalPipelineConfig without touching the Hub (no --policy.path on the CLI)."""
    argv = ["--env.type=pusht", "--policy.type=act", "--policy.device=cpu", *args]
    monkeypatch.setattr(sys, "argv", ["lerobot-eval", *argv])
    cfg = draccus.parse(EvalPipelineConfig, args=argv)
    if policy_path is not None:
        cfg.policy.pretrained_path = Path(policy_path)
    return cfg


def _info(video_paths=()):
    overall = {
        "avg_sum_reward": 1.5,
        "avg_max_reward": 0.9,
        "pc_success": 60.0,
        "n_episodes": 5,
        "n_success": 3,
        "pc_success_ci95": [23.1, 88.2],
        "eval_s": 12.0,
        "eval_ep_s": 2.4,
        "video_paths": list(video_paths),
        "predicted_video_paths": [],
    }
    return {"per_task": [], "per_group": {"pusht": dict(overall)}, "overall": overall}


def test_push_defaults_follow_the_job_target(monkeypatch):
    assert _cfg(monkeypatch).should_push_results is False
    assert _cfg(monkeypatch, "--job.target=a10g-small").should_push_results is True
    assert (
        _cfg(monkeypatch, "--job.target=a10g-small", "--push_results_to_hub=false").should_push_results
        is False
    )
    assert _cfg(monkeypatch, "--push_results_to_hub=true").should_push_results is True


def test_negative_max_videos_is_rejected(monkeypatch):
    # draccus wraps the config's ValueError.
    with pytest.raises(draccus.utils.ParsingError) as exc_info:
        _cfg(monkeypatch, "--results_max_videos=-1")
    assert "results_max_videos" in str(exc_info.value.__cause__)


def test_results_repo_defaults_to_the_hub_policy(monkeypatch):
    assert resolve_results_repo_id(_cfg(monkeypatch, policy_path="user/act_pusht")) == "user/act_pusht"
    cfg = _cfg(monkeypatch, "--results_repo_id=user/evals", policy_path="user/act_pusht")
    assert resolve_results_repo_id(cfg) == "user/evals"


def test_results_repo_required_for_a_local_policy(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="--results_repo_id"):
        resolve_results_repo_id(_cfg(monkeypatch, policy_path=tmp_path))
    cfg = _cfg(monkeypatch, "--results_repo_id=user/evals", policy_path=tmp_path)
    assert resolve_results_repo_id(cfg) == "user/evals"


def test_results_path_is_env_task_and_utc_timestamp(monkeypatch):
    cfg = _cfg(monkeypatch)
    assert results_slug(cfg) == "pusht-PushT-v0"
    assert results_path_in_repo(cfg, NOW) == "eval/pusht-PushT-v0/2026-11-02_09-30-05"
    # Non-UTC timestamps are converted, so paths sort consistently across machines.
    paris = NOW.astimezone(dt.timezone(dt.timedelta(hours=1)))
    assert results_path_in_repo(cfg, paris) == "eval/pusht-PushT-v0/2026-11-02_09-30-05"


def test_results_slug_sanitizes_multi_suite_tasks(monkeypatch):
    cfg = _cfg(monkeypatch)
    cfg.env.task = "libero_spatial,libero_object"
    assert results_slug(cfg) == "pusht-libero_spatial-libero_object"


def test_resolve_policy_revision(monkeypatch, tmp_path):
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(sha="abc123")
    cfg = _cfg(monkeypatch, policy_path="user/act_pusht")
    cfg.policy.pretrained_revision = "v1"

    assert resolve_policy_revision(cfg, api=api) == "abc123"
    api.model_info.assert_called_once_with("user/act_pusht", revision="v1")
    # A local policy has no Hub revision, and must not hit the Hub.
    assert resolve_policy_revision(_cfg(monkeypatch, policy_path=tmp_path), api=api) is None
    assert api.model_info.call_count == 1


def test_metadata_carries_provenance_and_summary(monkeypatch):
    cfg = _cfg(monkeypatch, "--eval.n_episodes=5", "--seed=7", policy_path="user/act_pusht")
    info = _info()
    info["overall"]["avg_max_reward"] = float("nan")

    meta = build_eval_metadata(cfg, info, "abc123", NOW)

    assert meta["format_version"] == 1
    assert meta["created_at"] == "2026-11-02T09:30:05+00:00"
    assert meta["lerobot_version"] == __version__
    assert meta["policy"] == {"repo_id": "user/act_pusht", "revision": "abc123", "type": "act"}
    assert meta["env"] == {"type": "pusht", "task": "PushT-v0"}
    assert meta["seed"] == 7
    assert meta["n_episodes_per_task"] == 5
    assert meta["summary"]["n_success"] == 3
    assert meta["summary"]["pc_success_ci95"] == [23.1, 88.2]
    assert meta["summary"]["avg_max_reward"] is None  # NaN is not valid JSON
    assert "video_paths" not in meta["summary"]
    assert "job" not in meta["config"]
    assert meta["config"]["env"]["type"] == "pusht"
    json.dumps(meta, allow_nan=False)


def test_push_uploads_one_commit_to_the_policy_repo(monkeypatch, tmp_path, caplog):
    videos = tmp_path / "videos" / "pusht_0"
    videos.mkdir(parents=True)
    paths = []
    for i in range(3):
        (videos / f"eval_episode_{i}.mp4").write_bytes(b"mp4")
        paths.append(str(videos / f"eval_episode_{i}.mp4"))
    cfg = _cfg(
        monkeypatch,
        f"--output_dir={tmp_path}",
        "--results_max_videos=2",
        "--push_results_to_hub=true",
        policy_path="user/act_pusht",
    )
    info = _info(paths)
    api = MagicMock()

    with caplog.at_level(logging.INFO):
        url = push_eval_results(cfg, info, policy_revision="abc123", api=api, now=NOW)

    folder = "eval/pusht-PushT-v0/2026-11-02_09-30-05"
    assert url == f"https://huggingface.co/user/act_pusht/tree/main/{folder}"
    # Pushing to the policy's own repo never creates it.
    api.create_repo.assert_not_called()
    api.create_commit.assert_called_once()
    kwargs = api.create_commit.call_args.kwargs
    assert kwargs["repo_id"] == "user/act_pusht"
    assert kwargs["repo_type"] == "model"
    assert "60.0% success (3/5 episodes)" in kwargs["commit_message"]
    ops = {op.path_in_repo: op for op in kwargs["operations"]}
    assert sorted(ops) == [
        f"{folder}/eval_info.json",
        f"{folder}/eval_metadata.json",
        f"{folder}/videos/pusht_0/eval_episode_0.mp4",
        f"{folder}/videos/pusht_0/eval_episode_1.mp4",
    ]
    # eval_info.json is uploaded unchanged, so `lerobot-eval-compare` reads the downloaded copy as-is.
    assert json.loads(ops[f"{folder}/eval_info.json"].path_or_fileobj) == info
    metadata = json.loads(ops[f"{folder}/eval_metadata.json"].path_or_fileobj)
    assert metadata["policy"]["revision"] == "abc123"
    # The HF Jobs submitter finishes following the job on this line.
    assert f"{RESULTS_PUSHED_MARKER}user/act_pusht/tree/main/{folder}" in caplog.text


def test_push_to_a_separate_repo_creates_it_private(monkeypatch):
    cfg = _cfg(monkeypatch, "--results_repo_id=me/evals", "--results_max_videos=0", policy_path="org/policy")
    api = MagicMock()

    push_eval_results(cfg, _info(["/nonexistent.mp4"]), policy_revision=None, api=api, now=NOW)

    api.create_repo.assert_called_once_with("me/evals", repo_type="model", private=True, exist_ok=True)
    assert api.create_commit.call_args.kwargs["repo_id"] == "me/evals"
    assert len(api.create_commit.call_args.kwargs["operations"]) == 2
