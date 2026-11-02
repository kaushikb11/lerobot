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
"""Run ``lerobot-eval`` on HF Jobs (HuggingFace GPUs).

Same shape as the training submitter in ``hf.py``. There is no config to stage:
the policy is already on the Hub, so the pod replays the user's own CLI flags
(minus the client-only ``--job.*``) and publishes the results to the Hub when
the evaluation ends, since the job's disk is discarded afterwards.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import is_dataclass
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, get_token, run_job

from lerobot.common.eval_results import RESULTS_PUSHED_MARKER, policy_hub_repo_id, resolve_results_repo_id

# Package-internal reuse of the training submitter's job plumbing: following a
# submitted job and forwarding argv are identical for evaluation runs.
from .hf import _pod_forwarded_args, follow_job, resolve_job_tags

if TYPE_CHECKING:
    from lerobot.configs.eval import EvalPipelineConfig

# Flags the submitter re-emits itself instead of forwarding verbatim. `--job.*` is
# dropped separately, by prefix; bare `--job` (a config file for the whole job
# section) is not, hence its entry here: it could smuggle a remote `target` onto
# the pod and have the job recursively submit itself.
_SUBMITTER_OWNED_ARGS = ("--push_results_to_hub", "--job")


def _local_config_file_args(cfg: EvalPipelineConfig) -> list[str]:
    """The CLI args that name a config file on the client's disk.

    draccus exposes ``--config_path`` for the whole config plus a ``--<field>``
    for every nested dataclass (``--env``, ``--eval``, ...). The pod has none of
    those files, so a remote run rejects them rather than silently dropping the
    settings they carry. ``--job`` is excluded: it only configures the submission.
    """
    nested = (name for name in vars(cfg) if name != "job" and is_dataclass(getattr(cfg, name)))
    return ["--config_path", *(f"--{name}" for name in nested)]


def build_pod_command(cfg: EvalPipelineConfig, argv: list[str]) -> list[str]:
    """Build the command the pod runs: ``lerobot-eval`` with the user's flags.

    ``--job.target=local`` stops the pod from re-dispatching to itself, and
    ``--push_results_to_hub`` is resolved here (on unless the user turned it off),
    so the pod never depends on the "on for jobs" default. With ``job.lerobot_ref``
    set, lerobot is first installed from git on top of the image's own copy.
    """
    forwarded = _pod_forwarded_args(argv, drop_names=_SUBMITTER_OWNED_ARGS, drop_prefixes=("--job.",))
    push = "true" if cfg.should_push_results else "false"
    command = ["lerobot-eval", *forwarded, "--job.target=local", f"--push_results_to_hub={push}"]
    if cfg.job.lerobot_ref is None:
        return command
    spec = f"lerobot @ git+{cfg.job.lerobot_repo}@{cfg.job.lerobot_ref}"
    # The image's venv is a uv venv without pip; the `python` on PATH is that venv's interpreter.
    # `--no-deps` keeps the image's pinned dependencies.
    install = (
        'uv pip install --python "$(command -v python)" --no-deps --reinstall-package lerobot '
        f"{shlex.quote(spec)}"
    )
    return ["bash", "-c", f"{install} && {shlex.join(command)}"]


def _check_can_publish(repo_id: str, user_info: dict) -> None:
    """Fail before spending GPU time when the results repo is clearly not writable.

    Only checks that the repo namespace is the user or one of their orgs: a
    fine-grained token can still lack write access, which then fails at upload.
    """
    namespace = repo_id.split("/", 1)[0] if "/" in repo_id else user_info["name"]
    namespaces = {user_info["name"], *(org["name"] for org in user_info.get("orgs", []))}
    if namespace not in namespaces:
        raise ValueError(
            f"Cannot publish eval results to {repo_id!r}: it is not under your account or one of your "
            f"organizations. Pass --results_repo_id={user_info['name']}/<repo> to publish them to your "
            "own repo, or --push_results_to_hub=false to only read the results in the job logs."
        )


def submit_eval_to_hf(cfg: EvalPipelineConfig) -> None:
    """Submit an evaluation run to HF Jobs infrastructure.

    Validates that the pod can load the policy (it must be on the Hub) and publish
    the results, submits the job, then tails its logs until the results are on the
    Hub or the job ends — or returns immediately with ``--job.detach``. Ctrl-C
    detaches without cancelling the remote job.
    """
    token = get_token()
    if not token:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")

    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError("Remote evaluation requires a pretrained policy: pass --policy.path=<user>/<repo>.")
    policy_repo_id = policy_hub_repo_id(cfg)
    if policy_repo_id is None:
        raise ValueError(
            f"--policy.path={cfg.policy.pretrained_path} is a local directory, which the HF Jobs pod "
            "cannot read. Upload it to the Hub first, e.g. "
            f"`hf upload <user>/<repo> {cfg.policy.pretrained_path} --repo-type=model`, "
            "then pass --policy.path=<user>/<repo>."
        )

    argv = sys.argv[1:]
    passed = {tok.split("=", 1)[0] for tok in argv}
    used_config_files = sorted(passed.intersection(_local_config_file_args(cfg)))
    if used_config_files:
        raise ValueError(
            f"{', '.join(used_config_files)} cannot be used with a remote --job.target: the pod "
            "cannot read config files from this machine. Pass the settings as CLI flags instead."
        )

    results_repo_id: str | None = None
    if cfg.should_push_results:
        results_repo_id = resolve_results_repo_id(cfg)
        api = HfApi(token=token)
        _check_can_publish(results_repo_id, api.whoami(token=token))
    else:
        print(
            "WARNING: --push_results_to_hub is off. The results live only on the job's disk, which is "
            "discarded when it ends; only the job logs will show them."
        )

    tags = resolve_job_tags(cfg.job.tags)
    command = build_pod_command(cfg, argv)

    print(f"Submitting eval job to HF Jobs (flavor={cfg.job.target}, image={cfg.job.image}) ...")
    job_info = run_job(
        image=cfg.job.image,
        command=command,
        flavor=cfg.job.target,
        secrets={"HF_TOKEN": token},
        timeout=cfg.job.timeout,
        # HF Jobs labels are key/value; expose each tag as a queryable label.
        labels=dict.fromkeys(tags, "true"),
    )
    job_id = job_info.id
    job_url = getattr(job_info, "url", None)
    print(f"Job submitted: {job_id}")
    if job_url:
        print(f"  Job page:     {job_url}")
    print(f"  Policy:       https://huggingface.co/{policy_repo_id}")
    if results_repo_id is not None:
        print(f"  Results repo: https://huggingface.co/{results_repo_id}")
    print(f"  Monitor:      hf jobs logs {job_id}")
    print(f"  Cancel:       hf jobs cancel {job_id}")

    # Finish as soon as the results are on the Hub (logged by `push_eval_results`), rather than
    # waiting out the platform's post-run finalization. Without a push, completion is stage-based.
    marker = f"{RESULTS_PUSHED_MARKER}{results_repo_id}" if results_repo_id is not None else None
    if not follow_job(job_id, detach=cfg.job.detach, success_marker=marker):
        return

    if results_repo_id is not None:
        print(f"\nEvaluation complete — results pushed to https://huggingface.co/{results_repo_id}")
    else:
        print("\nEvaluation complete. Note: --push_results_to_hub was off, so the results stayed on the pod.")
