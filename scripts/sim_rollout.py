"""Thin simulation rollout glue reusing native EpisodicStrategy and strict policy loading.

Keeps physical hardware wrappers (`scripts/rollout.py`) untouched: no physical homing,
READY/SpaceMouse guards, or `home_joints_deg` checks. Targets the RTC-compatible rollout
path (never silently falling back to synchronous inference); RTC scheduling parity with
the physical UR5e remains an open owner acceptance gate (see `docs/simulation-scope.md`).
"""

from typing import Any

from lerobot.configs import parser
from lerobot.rollout import context as rollout_context
from lerobot.rollout.strategies.episodic import EpisodicStrategy as NativeStrategy
from lerobot.scripts import lerobot_rollout as upstream

from scripts.policy_loading import load_rollout_policy
from scripts.rollout import RolloutConfig, bind_relative_anchor


class SimEpisodicStrategy(NativeStrategy):
    """Thin episodic rollout strategy for `SimRobot` with honest per-attempt accounting.

    Uses native `RolloutStrategy._start_episode` (which calls `SimRobot.start_episode()`,
    resets engine/interpolator/timer, and resumes inference) and native `_policy_loop`
    (which ends each episode on `episode_ended(robot.inner)` or `episode_time_s`).
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.attempts: list[dict[str, Any]] = []

    def _policy_loop(self, **kwargs):
        robot = kwargs["robot"]
        events = kwargs["events"]
        dataset = kwargs["dataset"]
        attempt_number = len(self.attempts) + 1
        episode_before = getattr(dataset, "num_episodes", 0)
        try:
            return super()._policy_loop(**kwargs)
        finally:
            inner = getattr(robot, "inner", robot)
            latest = getattr(inner, "latest", None)
            terminated = (
                bool(latest.terminated[0]) if latest is not None else False
            )
            truncated = (
                bool(latest.truncated[0]) if latest is not None else False
            )
            is_success = (
                bool(latest.is_success[0]) if latest is not None else False
            )
            elapsed_steps = (
                int(latest.elapsed_steps[0]) if latest is not None else 0
            )
            self.attempts.append(
                {
                    "attempt": attempt_number,
                    "dataset_episode_index": episode_before,
                    "elapsed_steps": elapsed_steps,
                    "terminated": terminated,
                    "truncated": truncated,
                    "is_success": is_success,
                    "discarded": bool(events.get("rerecord_episode", False)),
                    "stopped_early": bool(events.get("stop_recording", False)),
                }
            )


def _validate_sim_rollout_config(cfg: RolloutConfig) -> None:
    if cfg.robot is None or getattr(cfg.robot, "type", None) != "sim":
        raise ValueError(
            "Simulation rollout requires --robot.type=sim; use scripts.rollout for real hardware"
        )
    if cfg.teleop is not None:
        raise ValueError(
            "Simulation rollout is learned-policy-only and never accepts --teleop or a teacher"
        )
    if cfg.interactive or cfg.strategy.type != "episodic":
        raise ValueError(
            "Simulation rollout requires --strategy.type=episodic and --interactive=false"
        )
    if cfg.dataset is not None and cfg.dataset.push_to_hub:
        raise ValueError("Simulation datasets stay local: --dataset.push_to_hub must be false")
    inference_type = getattr(cfg.inference, "type", None)
    if inference_type != "rtc":
        raise ValueError(
            "Simulation rollout requires --inference.type=rtc; synchronous inference "
            f"(got {inference_type!r}) is not accepted without owner resolution"
        )
    rtc = getattr(cfg.inference, "rtc", None)
    if rtc is not None and not getattr(rtc, "enabled", True):
        raise ValueError(
            "RTC inference was requested with rtc.enabled=false; never silently fall back to sync"
        )


def run_sim_rollout(cfg: RolloutConfig):
    """Execute simulation rollout through native LeRobot episodic rollout and strict loader."""
    _validate_sim_rollout_config(cfg)

    cfg.strategy.reset_to_initial_position = False
    cfg.return_to_initial_position = False

    rtc = getattr(cfg.inference, "rtc", None)
    print(
        f"Sim rollout settings: robot={cfg.robot.type} env={getattr(cfg.robot.env, 'type', 'n/a')} "
        f"fps={cfg.fps:g} inference={cfg.inference.type} "
        f"rtc_enabled={getattr(rtc, 'enabled', 'n/a')} "
        f"rtc_execution_horizon={getattr(rtc, 'execution_horizon', 'n/a')} "
        "(NOTE: RTC physical-rollout parity remains an open owner decision in docs/simulation-scope.md — untested)",
        flush=True,
    )

    original_factory = upstream.create_strategy
    original_loader = rollout_context._load_pretrained_policy
    original_bind = rollout_context.bind_relative_anchor
    created: list[SimEpisodicStrategy] = []

    def create_strategy(config):
        if config.type == "episodic":
            strategy = SimEpisodicStrategy(config)
            created.append(strategy)
            return strategy
        return original_factory(config)

    upstream.create_strategy = create_strategy
    rollout_context._load_pretrained_policy = lambda config: load_rollout_policy(
        config, original_loader
    )
    rollout_context.bind_relative_anchor = bind_relative_anchor
    try:
        return upstream.rollout.__wrapped__(cfg)
    finally:
        upstream.create_strategy = original_factory
        rollout_context._load_pretrained_policy = original_loader
        rollout_context.bind_relative_anchor = original_bind
        if created and created[-1].attempts:
            attempts = created[-1].attempts
            successes = sum(1 for item in attempts if item["is_success"])
            print(
                f"Sim rollout attempts summary: {successes}/{len(attempts)} predicate successes "
                f"(verify every success against saved video): {attempts}",
                flush=True,
            )


@parser.wrap()
def sim_rollout(cfg: RolloutConfig):
    return run_sim_rollout(cfg)


if __name__ == "__main__":
    upstream.register_third_party_plugins()
    sim_rollout()
