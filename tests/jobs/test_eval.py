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

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import draccus
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.common.eval_results import RESULTS_PUSHED_MARKER
from lerobot.configs.eval import LEROBOT_GIT_URL, EvalJobConfig, EvalPipelineConfig
from lerobot.jobs.eval import build_pod_command, submit_eval_to_hf

pytestmark = pytest.mark.dataset

POLICY = "user/act_pusht"


def _cfg(monkeypatch, *args, policy_path=POLICY):
    """Parse an EvalPipelineConfig from a CLI, as `lerobot-eval` would, without touching the Hub.

    `--policy.type` stands in for `--policy.path` (which would download config.json); the Hub
    path is then set the way `EvalPipelineConfig.__post_init__` sets it.
    """
    argv = ["--env.type=pusht", "--policy.type=act", *args]
    monkeypatch.setattr(sys, "argv", ["lerobot-eval", *argv])
    cfg = draccus.parse(EvalPipelineConfig, args=argv)
    if policy_path is not None:
        cfg.policy.pretrained_path = Path(policy_path)
    return cfg


@pytest.fixture
def hub(monkeypatch):
    """Mock every Hub/Jobs call the submitter makes and record them."""
    calls = SimpleNamespace(run_job=[], follow_job=[])
    api = MagicMock()
    api.whoami.return_value = {"name": "user", "orgs": [{"name": "my-org"}]}
    monkeypatch.setattr("lerobot.jobs.eval.get_token", lambda: "hf_token")
    monkeypatch.setattr("lerobot.jobs.eval.HfApi", lambda token=None: api)

    def _run_job(**kwargs):
        calls.run_job.append(kwargs)
        return SimpleNamespace(id="job-1", url="https://huggingface.co/jobs/user/job-1")

    def _follow_job(job_id, *, detach, success_marker=None):
        calls.follow_job.append({"job_id": job_id, "detach": detach, "success_marker": success_marker})
        return not detach

    monkeypatch.setattr("lerobot.jobs.eval.run_job", _run_job)
    monkeypatch.setattr("lerobot.jobs.eval.follow_job", _follow_job)
    calls.api = api
    return calls


# --- config ----------------------------------------------------------------


def test_eval_job_defaults():
    job = EvalJobConfig()
    assert job.is_remote is False
    assert job.image == "huggingface/lerobot-gpu:latest"
    assert job.timeout == "6h"
    assert job.lerobot_ref is None
    assert job.lerobot_repo == LEROBOT_GIT_URL


def test_eval_config_parses_job_flags(monkeypatch):
    cfg = _cfg(monkeypatch, "--job.target", "a10g-small", "--job.timeout=1h", "--job.detach=true")
    assert cfg.job.is_remote is True
    assert (cfg.job.target, cfg.job.timeout, cfg.job.detach) == ("a10g-small", "1h", True)


# --- pod command -----------------------------------------------------------


def test_pod_command_replays_cli_without_job_flags(monkeypatch):
    cfg = _cfg(
        monkeypatch,
        "--eval.n_episodes=5",
        "--job.target",
        "a10g-small",
        "--job.tags=[nightly]",
        "--push_results_to_hub",
        "true",
    )
    command = build_pod_command(cfg, sys.argv[1:])
    assert command[0] == "lerobot-eval"
    assert "--eval.n_episodes=5" in command
    assert not [a for a in command if a.startswith("--job.") and a != "--job.target=local"]
    assert "a10g-small" not in command
    # The submitter's own flags close the command: never re-dispatch, and push explicitly.
    assert command[-2:] == ["--job.target=local", "--push_results_to_hub=true"]
    assert command.count("true") == 0  # the space-separated value was dropped with its flag


def test_pod_command_keeps_an_explicit_push_opt_out(monkeypatch):
    cfg = _cfg(monkeypatch, "--job.target=t4-small", "--push_results_to_hub=false")
    assert build_pod_command(cfg, sys.argv[1:])[-1] == "--push_results_to_hub=false"


def test_pod_command_installs_lerobot_ref(monkeypatch):
    cfg = _cfg(
        monkeypatch,
        "--job.target=t4-small",
        "--job.lerobot_ref=feat/eval-hf-jobs",
        "--job.lerobot_repo=https://github.com/me/lerobot.git",
    )
    command = build_pod_command(cfg, sys.argv[1:])
    assert command[:2] == ["bash", "-c"]
    install, _, run = command[2].partition(" && ")
    assert "uv pip install" in install
    assert "--no-deps" in install
    assert shlex.quote("lerobot @ git+https://github.com/me/lerobot.git@feat/eval-hf-jobs") in install
    run_argv = shlex.split(run)
    assert run_argv[0] == "lerobot-eval"
    assert run_argv[-2:] == ["--job.target=local", "--push_results_to_hub=true"]


# --- submit ----------------------------------------------------------------


def test_submit_runs_eval_on_the_pod_and_waits_for_the_results(monkeypatch, hub):
    cfg = _cfg(monkeypatch, "--job.target=a10g-small", "--job.tags=[nightly]", "--eval.n_episodes=5")

    submit_eval_to_hf(cfg)

    (job,) = hub.run_job
    assert job["flavor"] == "a10g-small"
    assert job["image"] == "huggingface/lerobot-gpu:latest"
    assert job["timeout"] == "6h"
    assert job["secrets"] == {"HF_TOKEN": "hf_token"}
    assert job["labels"] == {"lerobot": "true", "nightly": "true"}
    assert job["command"][0] == "lerobot-eval"
    assert "--eval.n_episodes=5" in job["command"]
    assert job["command"][-1] == "--push_results_to_hub=true"
    assert hub.follow_job == [
        {"job_id": "job-1", "detach": False, "success_marker": f"{RESULTS_PUSHED_MARKER}{POLICY}"}
    ]


def test_submit_without_push_follows_until_the_job_ends(monkeypatch, hub, capsys):
    cfg = _cfg(monkeypatch, "--job.target=a10g-small", "--push_results_to_hub=false")

    submit_eval_to_hf(cfg)

    assert "--push_results_to_hub is off" in capsys.readouterr().out
    assert hub.follow_job[0]["success_marker"] is None
    hub.api.whoami.assert_not_called()


def test_submit_detach_returns_after_submitting(monkeypatch, hub, capsys):
    cfg = _cfg(monkeypatch, "--job.target=a10g-small", "--job.detach=true")

    submit_eval_to_hf(cfg)

    assert len(hub.run_job) == 1
    assert hub.follow_job[0]["detach"] is True
    assert "Evaluation complete" not in capsys.readouterr().out


def test_submit_rejects_a_local_policy(monkeypatch, hub, tmp_path):
    cfg = _cfg(monkeypatch, "--job.target=a10g-small", policy_path=tmp_path)
    with pytest.raises(ValueError, match="hf upload"):
        submit_eval_to_hf(cfg)
    assert hub.run_job == []


def test_submit_rejects_a_policy_from_scratch(monkeypatch, hub):
    cfg = _cfg(monkeypatch, "--job.target=a10g-small", policy_path=None)
    with pytest.raises(ValueError, match="--policy.path"):
        submit_eval_to_hf(cfg)
    assert hub.run_job == []


def test_submit_rejects_local_config_files(monkeypatch, hub):
    cfg = _cfg(monkeypatch, "--job.target=a10g-small")
    # Parsing would read the file; the submitter only sees it on the raw CLI.
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--eval=my_eval.yaml"])
    with pytest.raises(ValueError, match="--eval cannot be used"):
        submit_eval_to_hf(cfg)
    assert hub.run_job == []


def test_submit_rejects_results_repo_outside_the_users_namespaces(monkeypatch, hub):
    cfg = _cfg(monkeypatch, "--job.target=a10g-small", policy_path="lerobot/diffusion_pusht")
    with pytest.raises(ValueError, match="--results_repo_id=user/"):
        submit_eval_to_hf(cfg)
    assert hub.run_job == []


@pytest.mark.parametrize("results_repo", ["user/evals", "my-org/evals"])
def test_submit_publishes_a_foreign_policy_to_an_owned_repo(monkeypatch, hub, results_repo):
    cfg = _cfg(
        monkeypatch,
        "--job.target=a10g-small",
        f"--results_repo_id={results_repo}",
        policy_path="lerobot/diffusion_pusht",
    )
    submit_eval_to_hf(cfg)
    assert f"--results_repo_id={results_repo}" in hub.run_job[0]["command"]
    assert hub.follow_job[0]["success_marker"] == f"{RESULTS_PUSHED_MARKER}{results_repo}"


def test_submit_requires_login(monkeypatch, hub):
    monkeypatch.setattr("lerobot.jobs.eval.get_token", lambda: None)
    with pytest.raises(RuntimeError, match="hf auth login"):
        submit_eval_to_hf(_cfg(monkeypatch, "--job.target=a10g-small"))


# --- lerobot-eval dispatch -------------------------------------------------


def test_eval_main_dispatches_remote_runs_to_the_submitter(monkeypatch):
    import lerobot.scripts.lerobot_eval as eval_module

    captured = []
    monkeypatch.setattr(eval_module, "submit_eval_to_hf", captured.append)

    def _no_local_eval(*args, **kwargs):
        raise AssertionError("a remote run must not build envs locally")

    monkeypatch.setattr(eval_module, "make_env", _no_local_eval)
    cfg = _cfg(monkeypatch, "--job.target=a10g-small")

    eval_module.eval_main(cfg)

    assert captured == [cfg]
