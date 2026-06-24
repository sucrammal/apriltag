import os
import datetime
import asyncio
import dt_apriltags as apriltag
import numpy as np
import cv2
import math

from scipy.spatial.transform import Rotation
from .spatialmath import quaternion_to_orientation_vector

from typing import (Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, cast)
from typing_extensions import Self

from viam.components.pose_tracker import PoseTracker
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, ViamImage
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, PoseInFrame, Pose, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.registry import Registry
from viam.resource.types import Model, ModelFamily, RESOURCE_TYPE_COMPONENT, RESOURCE_TYPE_SERVICE
from viam.errors import ResourceNotFoundError
from viam.logging import getLogger
from viam.utils import struct_to_dict, ValueTypes
from viam.media.utils.pil import viam_to_pil_image
from PIL import Image


# required attributes
cam_attr = "camera_name"
family_attr = "tag_family"
width_attr = "tag_width_mm"

LOGGER = getLogger(__name__)


class ApriltagModule(Module):
    """Module wrapper that resolves dependencies without a full parent refresh.

    The default Module._get_resource calls parent.refresh(), which tries to
    remove every resource that was disabled in the machine config. On older
    viam-sdk versions that can KeyError for resources this module never cached
    (for example an unrelated disabled gripper), causing apriltag reconfigure to
    fail even though it only depends on its camera.
    """

    async def _get_resource(self, name: ResourceName) -> ResourceBase:
        await self._connect_to_parent()
        assert self.parent is not None

        if name.type == RESOURCE_TYPE_COMPONENT:
            getter = self.parent.get_component
        elif name.type == RESOURCE_TYPE_SERVICE:
            getter = self.parent.get_service
        else:
            raise ValueError("Dependency does not describe a component nor a service")

        try:
            return getter(name)
        except ResourceNotFoundError:
            await self.parent._create_or_reset_client(name)
            return getter(name)


class Apriltag(PoseTracker, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("marcus-org", "apriltag"), "pose_tracker")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        instance = super().new(config, dependencies)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        cam = attrs.get(cam_attr)
        if cam is None:
            raise Exception("Missing required " + cam_attr + " attribute.")
        if attrs.get(family_attr) is None:
            raise Exception("Missing requried " + family_attr + " attribute.")
        if attrs.get(width_attr) is None:
            raise Exception("Missing requried " + width_attr + " attribute.")
        return [str(cam)], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        attrs = struct_to_dict(config.attributes)
        cam_name = str(attrs.get(cam_attr))
        self.camera = cast(Camera, dependencies[Camera.get_resource_name(cam_name)])
        self.tag_family = attrs.get(family_attr)
        self.tag_width_mm = attrs.get(width_attr)

    async def get_poses(
        self,
        body_names: List[str],
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, PoseInFrame]:
        """This method returns the poses of the requested Apriltag IDs. 
        If no body names are requested, all detected Apriltags are returned

        Args:
            body_names (List[str]): A list of Apriltag IDs to return

        Returns:
            Dict[str, PoseInFrame]: A dictionary mapping Apriltag ID strings to their detected PoseInFrame
        """
        try:
            # need to get the camera intrinsics
            properties = await self.camera.get_properties()
            intrinsics = [
                properties.intrinsic_parameters.focal_x_px,
                properties.intrinsic_parameters.focal_y_px,
                properties.intrinsic_parameters.center_x_px,
                properties.intrinsic_parameters.center_y_px
            ]

            # get an image from camera resource and convert it to OpenCV format
            cam_images = await self.camera.get_images()
            gray_image = None
            color_image = None
            for image in cam_images[0]:
                if image.mime_type == CameraMimeType.JPEG:
                    color_image = cam_images[0][0].data
                    gray_image = cv2.cvtColor(np.array(viam_to_pil_image(cam_images[0][0])), cv2.COLOR_RGB2GRAY)  # convert to grayscale

            if gray_image is None or color_image is None:
                raise Exception("camera had no jpeg images")
                

            # initialize AprilTag detector - can include multiple families of tags in comma separated string
            detector = apriltag.Detector(families=self.tag_family)
            tags = detector.detect(gray_image, estimate_tag_pose=True, camera_params=intrinsics, tag_size=0.001*self.tag_width_mm)

            poses = {}
            for tag in tags:
                #
                if len(body_names) == 0 or str(tag.tag_id) in body_names:
                    o = quaternion_to_orientation_vector(Rotation.from_matrix(tag.pose_R))
                    # need to convert the returned positions from mm to m and the orientation's theta to degrees.
                    poses[str(tag.tag_id)] = PoseInFrame(
                        reference_frame=self.camera.name, 
                        pose=Pose(
                            x=tag.pose_t[0][0] * 1000,
                            y=tag.pose_t[1][0] * 1000,
                            z=tag.pose_t[2][0] * 1000, 
                            o_x=o.o_x,
                            o_y=o.o_y,
                            o_z=o.o_z,
                            theta=o.theta * 180 / math.pi
                        )
                    )        
            time = datetime.datetime.utcnow().isoformat() + "Z"
            viam_home = os.getenv("VIAM_HOME")
            if viam_home is None:
                raise Exception("VIAM_HOME not set")

            capturedir = os.path.join(viam_home,"capture")
            root_path = os.path.join(capturedir,self.name,time)
            os.makedirs(root_path)
            with open(os.path.join(root_path, "./color_image.jpeg"), 'wb') as f:
                f.write(color_image)
            im = Image.fromarray(gray_image)
            im.save(os.path.join(root_path, "./gray_image.jpeg"))
            return poses
                
        except Exception as e:
            raise e
        
    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> List[Geometry]:
        raise NotImplementedError()

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()


class ApriltagCamera(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("marcus-org", "apriltag"), "camera")

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        cam = attrs.get(cam_attr)
        if cam is None:
            raise Exception("Missing required " + cam_attr + " attribute.")
        if attrs.get(family_attr) is None:
            raise Exception("Missing required " + family_attr + " attribute.")
        if attrs.get(width_attr) is None:
            raise Exception("Missing required " + width_attr + " attribute.")
        return [str(cam)], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        attrs = struct_to_dict(config.attributes)
        cam_name = str(attrs.get(cam_attr))
        self.camera = cast(Camera, dependencies[Camera.get_resource_name(cam_name)])
        self.tag_family = attrs.get(family_attr)
        self.tag_width_mm = attrs.get(width_attr)

    async def get_image(
        self,
        mime_type: str = "",
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ViamImage:
        properties = await self.camera.get_properties()
        intrinsics = [
            properties.intrinsic_parameters.focal_x_px,
            properties.intrinsic_parameters.focal_y_px,
            properties.intrinsic_parameters.center_x_px,
            properties.intrinsic_parameters.center_y_px,
        ]

        cam_images, _ = await self.camera.get_images()
        source = None
        for img in cam_images:
            if img.mime_type == CameraMimeType.JPEG:
                source = img
                break
        if source is None:
            raise Exception("camera had no JPEG images")

        pil_img = viam_to_pil_image(source)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        detector = apriltag.Detector(families=self.tag_family)
        tags = detector.detect(
            gray, estimate_tag_pose=True, camera_params=intrinsics, tag_size=0.001 * self.tag_width_mm
        )

        for tag in tags:
            corners = tag.corners.astype(int)
            cv2.polylines(bgr, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            center = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(bgr, center, 5, (0, 0, 255), -1)
            cv2.putText(bgr, f"ID:{tag.tag_id}", (center[0] + 8, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, jpeg_bytes = cv2.imencode(".jpg", bgr)
        return ViamImage(jpeg_bytes.tobytes(), CameraMimeType.JPEG)

    async def get_images(self, *, extra=None, timeout=None, **kwargs):
        img = await self.get_image(timeout=timeout, extra=extra)
        return [img], None

    async def get_properties(self, *, extra=None, timeout=None, **kwargs):
        return await self.camera.get_properties()

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError()

    async def get_geometries(self, *, extra=None, timeout=None):
        raise NotImplementedError()


async def run_module():
    module = ApriltagModule.from_args()
    for key in Registry.REGISTERED_RESOURCE_CREATORS().keys():
        module.add_model_from_registry(*key.split("/"))  # pyright: ignore [reportArgumentType]
    await module.start()


if __name__ == "__main__":
    asyncio.run(run_module())
