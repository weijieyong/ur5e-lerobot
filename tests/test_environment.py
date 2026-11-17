"""Python import isolation from desktop ROS/user environments; no hardware."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_rollout_cli_uses_project_package_when_ros_scripts_package_is_present(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "scripts"
            package.mkdir()
            (package / "__init__.py").write_text(
                'raise RuntimeError("external ROS scripts package selected")\n'
            )
            env = dict(
                os.environ,
                PYTHONPATH=directory
                + os.pathsep
                + str(ROOT / "src/lerobot_robot_ur5e"),
            )
            result = subprocess.run(
                [sys.executable, "-m", "scripts.rollout", "--help"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("external ROS scripts package selected", result.stderr)
            self.assertIn("--robot.type", result.stdout)

    def test_activation_removes_python_overrides_before_starting_pixi_python(self):
        pixi = os.environ.get("PIXI_EXE") or shutil.which("pixi")
        if pixi is None:
            self.skipTest(
                "Run through pixi run test for the Pixi child-environment check"
            )
        env = dict(
            os.environ,
            PIXI_PROJECT_ROOT=str(ROOT),
            PYTHONPATH="/external/ros/python",
            PYTHONHOME="/invalid/external/python",
            PYTHONNOUSERSITE="0",
            LD_LIBRARY_PATH="/external/ros/lib:/usr/local/cuda-11.8/lib64",
        )
        check = (
            "import os,sys,site; "
            "assert not os.environ.get('PYTHONPATH'); "
            "assert not os.environ.get('PYTHONHOME'); "
            "assert not os.environ.get('LD_LIBRARY_PATH'); "
            "assert sys.flags.no_user_site == 1 and not site.ENABLE_USER_SITE"
        )
        # Reuse the parent's selected workspace target for CPU-only checks on
        # hosts where CUDA virtual-package detection is unavailable.
        platforms = os.environ.get("PIXI_ENVIRONMENT_PLATFORMS", "").split(",")
        target = (
            ["--platform", platforms[0]] if len(platforms) == 1 and platforms[0] else []
        )
        result = subprocess.run(
            [pixi, "run", "--frozen", *target, "python", "-c", check],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
