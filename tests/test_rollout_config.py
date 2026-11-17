"""Exercise the native CLI/config boundary without network, models or hardware."""

import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lerobot.configs import parser
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.robots import RobotConfig
from scripts import rollout


@RobotConfig.register_subclass("review_config_only")
@dataclass
class ConfigOnlyRobot(RobotConfig):
    pass


class RolloutConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "checkpoint"
        ACTConfig(device="cpu").save_pretrained(self.checkpoint)
        # Simulate a cached non-PI05 config: offline environment alone cannot
        # prevent this config from substituting for a missing local deployment.
        self.hub = self.enterContext(
            patch(
                "lerobot.configs.policies.hf_hub_download",
                return_value=str(self.checkpoint / "config.json"),
            )
        )
        self.run = self.enterContext(
            patch.object(rollout, "run_rollout", side_effect=lambda cfg: cfg)
        )
        self.hardware = self.enterContext(
            patch.object(
                rollout.upstream.rollout,
                "__wrapped__",
                side_effect=AssertionError("model/hardware entry reached"),
            )
        )
        self.enterContext(
            patch("socket.socket.connect", side_effect=AssertionError("network"))
        )
        self.enterContext(patch.dict(parser._config_path_args, {}, clear=True))
        self.enterContext(patch.dict(parser._config_yaml_overrides, {}, clear=True))

    def invoke(self, args):
        with patch.object(sys, "argv", ["rollout", *args]):
            return rollout.rollout()

    def cli_args(self, path, separated=False):
        option = ["--policy.path", path] if separated else [f"--policy.path={path}"]
        return ["--robot.type=review_config_only", "--device=cpu", *option]

    def assert_rejected(self, args):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            with self.assertRaises((FileNotFoundError, SystemExit)) as caught:
                self.invoke(args)
        self.assertIn(
            "checkpoint directory is missing", str(caught.exception) + errors.getvalue()
        )
        self.hub.assert_not_called()
        self.run.assert_not_called()
        self.hardware.assert_not_called()

    def test_missing_local_cli_paths_fail_before_native_hub_config_resolution(self):
        for path in (
            "outputs/missing",
            "./namespace/missing",
            str(self.root / "missing"),
        ):
            for separated in (False, True):
                with self.subTest(path=path, separated=separated):
                    self.assert_rejected(self.cli_args(path, separated))

    def config_args(self, path, suffix):
        config = self.root / f"rollout{suffix}"
        # JSON is also valid YAML; exercise each native file dispatch.
        config.write_text(
            json.dumps(
                {
                    "robot": {"type": "review_config_only"},
                    "device": "cpu",
                    "policy": {"path": path},
                }
            )
        )
        return ["--config_path", str(config)]

    def test_missing_local_yaml_and_json_paths_fail_before_hub_resolution(self):
        for suffix in (".yaml", ".yml", ".json"):
            with self.subTest(suffix=suffix):
                self.assert_rejected(self.config_args("outputs/missing", suffix))

    def test_valid_local_config_uses_native_resolution(self):
        for args in (
            self.cli_args(str(self.checkpoint)),
            self.cli_args(str(self.checkpoint), separated=True),
            self.config_args(str(self.checkpoint), ".yaml"),
            self.config_args(str(self.checkpoint), ".json"),
        ):
            with self.subTest(args=args):
                cfg = self.invoke(args)
                self.assertIsInstance(cfg, rollout.upstream.RolloutConfig)
                self.assertIsInstance(cfg.policy, ACTConfig)
                self.assertEqual(cfg.policy.pretrained_path, str(self.checkpoint))
        self.hub.assert_not_called()
        self.hardware.assert_not_called()

    def test_intentional_hub_strings_use_native_resolution(self):
        for identifier in ("namespace/model", "model"):
            for separated in (False, True):
                with self.subTest(identifier=identifier, separated=separated):
                    cfg = self.invoke(self.cli_args(identifier, separated))
                    self.assertIsInstance(cfg.policy, ACTConfig)
                    self.assertEqual(cfg.policy.pretrained_path, identifier)
                    self.assertEqual(self.hub.call_args.kwargs["repo_id"], identifier)
        self.hardware.assert_not_called()

    def test_cli_path_overrides_yaml_path(self):
        cfg = self.invoke(
            self.config_args("outputs/missing", ".yaml")
            + ["--policy.path=namespace/model"]
        )
        self.assertEqual(cfg.policy.pretrained_path, "namespace/model")
        self.hub.assert_called_once()

    def test_help_does_not_resolve_policy_or_enter_rollout(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self.invoke(["--policy.path=outputs/missing", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.hub.assert_not_called()
        self.run.assert_not_called()
        self.hardware.assert_not_called()

    def test_programmatic_path_objects_are_local(self):
        config_class = rollout.rollout.__annotations__["cfg"]
        with patch.object(sys, "argv", ["rollout"]):
            for value in (Path("./namespace/missing"), Path("missing")):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(
                        FileNotFoundError, "checkpoint directory is missing"
                    ):
                        config_class(
                            robot=ConfigOnlyRobot(),
                            policy=ACTConfig(device="cpu", pretrained_path=value),
                        )
            cfg = config_class(
                robot=ConfigOnlyRobot(),
                policy=ACTConfig(device="cpu", pretrained_path=self.checkpoint),
            )
        self.assertEqual(cfg.policy.pretrained_path, self.checkpoint)
        self.hub.assert_not_called()


if __name__ == "__main__":
    unittest.main()
