"""LeRobot camera construction with manual controls for RGB-only D405 capture."""

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs


class RGBRealSenseCamera(RealSenseCamera):
    def _get_color_sensor(self):
        # LeRobot 0.7.0 still requires a dedicated RGB sensor for manual controls.
        # Our D405 captures RGB only, so selecting its shared Stereo Module is intentional.
        if self.rs_profile is not None and self.use_rgb and not self.use_depth:
            import pyrealsense2 as rs

            device = self.rs_profile.get_device()
            if "D405" in device.get_info(rs.camera_info.name):
                return device.first_depth_sensor()
        return super()._get_color_sensor()


def make_cameras(configs):
    return {
        name: RGBRealSenseCamera(config)
        if isinstance(config, RealSenseCameraConfig)
        else make_cameras_from_configs({name: config})[name]
        for name, config in configs.items()
    }
