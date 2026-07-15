import os
import datetime
import asyncio
import dt_apriltags as apriltag
import numpy as np
import cv2
import math

from scipy.spatial.transform import Rotation
from .spatialmath import quaternion_to_orientation_vector
from .pcd_utils import (
    filter_points_in_bbox,
    read_pcd_xyz,
    segment_geometry,
    write_pcd_xyz,
)

from typing import (Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, cast)
from typing_extensions import Self

from viam.components.pose_tracker import PoseTracker
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.common import ResponseMetadata
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, GeometriesInFrame, PointCloudObject, PoseInFrame, Pose, ResourceName
from viam.proto.service.vision import Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.registry import Registry
from viam.resource.types import Model, ModelFamily, RESOURCE_TYPE_COMPONENT, RESOURCE_TYPE_SERVICE
from viam.errors import ResourceNotFoundError
from viam.logging import getLogger
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import struct_to_dict, ValueTypes
from viam.media.utils.pil import viam_to_pil_image
from PIL import Image


# required attributes
cam_attr = "camera_name"
family_attr = "tag_family"
width_attr = "tag_width_mm"
confidence_threshold_attr = "confidence_threshold_pct"
bbox_padding_attr = "bbox_padding_px"
min_segment_points_attr = "min_segment_points"

LOGGER = getLogger(__name__)


def _color_image_from_camera_images(
    cam_images: Sequence[NamedImage],
) -> NamedImage:
    """Pick the color frame from a camera GetImages response (Orbbec, crop-camera, etc.)."""
    if not cam_images:
        raise Exception("camera returned no images")

    def is_depth(img: NamedImage) -> bool:
        name = (img.name or "").lower()
        if "depth" in name:
            return True
        return img.mime_type in (CameraMimeType.VIAM_RAW_DEPTH, CameraMimeType.PCD)

    by_name = {(img.name or "").lower(): img for img in cam_images}
    if "color" in by_name and not is_depth(by_name["color"]):
        return by_name["color"]

    jpeg = next(
        (img for img in cam_images if img.mime_type == CameraMimeType.JPEG and not is_depth(img)),
        None,
    )
    if jpeg is not None:
        return jpeg

    color = next((img for img in cam_images if not is_depth(img)), None)
    if color is not None:
        return color

    raise Exception("camera returned no color images")


def _gray_from_viam_image(image: ViamImage) -> tuple[np.ndarray, int, int]:
    pil_img = viam_to_pil_image(image)
    rgb = np.array(pil_img)
    if rgb.ndim == 2:
        gray = rgb
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    width = int(image.width or rgb.shape[1])
    height = int(image.height or rgb.shape[0])
    return gray, width, height


def _detect_apriltags(gray_image: np.ndarray, tag_family: str) -> list[Any]:
    detector = apriltag.Detector(families=tag_family)
    return list(detector.detect(gray_image))


def _tag_confidence(tag: Any) -> float:
    margin = getattr(tag, "decision_margin", None)
    if margin is None:
        return 1.0
    # decision_margin is commonly 20-150; map so good tags clear segmenter defaults.
    return float(min(max(margin / 40.0, 0.0), 1.0))


def _parse_optional_float(attrs: Mapping[str, Any], key: str, default: float) -> float:
    value = attrs.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise Exception(f"{key} must be a number")
    return float(value)


def _parse_optional_int(attrs: Mapping[str, Any], key: str, default: int) -> int:
    value = attrs.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise Exception(f"{key} must be an integer")
    return int(value)


def _expand_bbox(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    padding_px: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if padding_px <= 0:
        return x_min, y_min, x_max, y_max
    return (
        max(0, x_min - padding_px),
        max(0, y_min - padding_px),
        min(width - 1, x_max + padding_px),
        min(height - 1, y_max + padding_px),
    )


def _parse_confidence_threshold(attrs: Mapping[str, Any]) -> float:
    threshold = attrs.get(confidence_threshold_attr, 0.0)
    if threshold is None:
        return 0.0
    if not isinstance(threshold, (int, float)):
        raise Exception(confidence_threshold_attr + " must be a number between 0.0 and 1.0")
    threshold = float(threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise Exception(confidence_threshold_attr + " must be between 0.0 and 1.0")
    return threshold


def _tags_to_detections(
    tags: Sequence[Any],
    width: int,
    height: int,
    *,
    confidence_threshold_pct: float = 0.0,
    bbox_padding_px: int = 0,
) -> List[Detection]:
    if width <= 0 or height <= 0:
        raise Exception("image width and height are required for detections")

    detections: List[Detection] = []
    for tag in tags:
        confidence = _tag_confidence(tag)
        if confidence < confidence_threshold_pct:
            continue
        xs = tag.corners[:, 0]
        ys = tag.corners[:, 1]
        # Detection bbox fields are int64 in the vision proto, not float.
        x_min = int(round(float(np.min(xs))))
        y_min = int(round(float(np.min(ys))))
        x_max = int(round(float(np.max(xs))))
        y_max = int(round(float(np.max(ys))))
        x_min, y_min, x_max, y_max = _expand_bbox(
            x_min, y_min, x_max, y_max, bbox_padding_px, width, height
        )
        detections.append(
            Detection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                x_min_normalized=x_min / width,
                y_min_normalized=y_min / height,
                x_max_normalized=x_max / width,
                y_max_normalized=y_max / height,
                confidence=confidence,
                class_name=str(tag.tag_id),
            )
        )
    return detections


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
            time = datetime.datetime.now(datetime.timezone.utc).isoformat()
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
        cmd = dict(command)
        if "get_poses" in cmd:
            body_names = cmd["get_poses"] if isinstance(cmd["get_poses"], list) else []
            poses = await self.get_poses(body_names, timeout=timeout)
            return {
                tag_id: {
                    "x": p.pose.x,
                    "y": p.pose.y,
                    "z": p.pose.z,
                    "o_x": p.pose.o_x,
                    "o_y": p.pose.o_y,
                    "o_z": p.pose.o_z,
                    "theta": p.pose.theta,
                    "reference_frame": p.reference_frame,
                }
                for tag_id, p in poses.items()
            }
        raise NotImplementedError(f"unknown command: {list(cmd.keys())}")


class ApriltagCamera(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("marcus-org", "apriltag"), "camera")

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

    async def get_images(
        self,
        *,
        filter_source_names: Optional[Sequence[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Sequence[NamedImage], ResponseMetadata]:
        try:
            cam_images, metadata = await self.camera.get_images(timeout=timeout)
        except Exception as e:
            LOGGER.error("ApriltagCamera.get_images: failed to get images from source camera: %s", e)
            raise

        source = next((img for img in cam_images if img.mime_type == CameraMimeType.JPEG), None)
        if source is None and cam_images:
            source = cam_images[0]
        if source is None:
            raise Exception("source camera returned no images")

        pil_img = viam_to_pil_image(source)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        tags = _detect_apriltags(gray, self.tag_family)

        for tag in tags:
            corners = tag.corners.astype(int)
            cv2.polylines(bgr, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            center = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(bgr, center, 5, (0, 0, 255), -1)
            cv2.putText(bgr, f"ID:{tag.tag_id}", (center[0] + 8, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, jpeg_bytes = cv2.imencode(".jpg", bgr)
        named = NamedImage(name=self.name, data=jpeg_bytes.tobytes(), mime_type=CameraMimeType.JPEG)
        return [named], metadata

    async def get_properties(self, *, extra=None, timeout=None, **kwargs):
        return await self.camera.get_properties()

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError()

    async def get_geometries(self, *, extra=None, timeout=None):
        raise NotImplementedError()


class ApriltagVision(Vision, EasyResource):
    """2D AprilTag detector for use with viam:vision:detections-to-segments."""

    MODEL: ClassVar[Model] = Model(ModelFamily("marcus-org", "apriltag"), "vision")

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
            raise Exception("Missing required " + family_attr + " attribute.")
        _parse_confidence_threshold(attrs)
        _parse_optional_int(attrs, bbox_padding_attr, 40)
        _parse_optional_int(attrs, min_segment_points_attr, 3)
        return [str(cam)], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        attrs = struct_to_dict(config.attributes)
        cam_name = str(attrs.get(cam_attr))
        self.camera = cast(Camera, dependencies[Camera.get_resource_name(cam_name)])
        self.tag_family = attrs.get(family_attr)
        self.confidence_threshold_pct = _parse_confidence_threshold(attrs)
        # Padded boxes for GetDetections (used by detections-to-segments).
        self.bbox_padding_px = _parse_optional_int(attrs, bbox_padding_attr, 40)
        self.min_segment_points = _parse_optional_int(attrs, min_segment_points_attr, 3)

    def _camera_for_request(self, camera_name: str) -> Camera:
        if camera_name != self.camera.name:
            raise Exception(
                f"this detector is configured for camera {self.camera.name!r}, "
                f"got request for {camera_name!r}. "
                f"When using detections-to-segments with crop-camera, set camera_name to "
                f"'crop-camera' on the detector too."
            )
        return self.camera

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=True,
        )

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        camera = self._camera_for_request(camera_name)
        cam_images, _ = await camera.get_images(timeout=timeout)
        source = _color_image_from_camera_images(cam_images)
        gray, width, height = _gray_from_viam_image(source)
        tags = _detect_apriltags(gray, self.tag_family)
        return _tags_to_detections(
            tags,
            width,
            height,
            confidence_threshold_pct=self.confidence_threshold_pct,
            bbox_padding_px=0,
        )

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        """Return detections with padded bboxes for detections-to-segments."""
        gray, width, height = _gray_from_viam_image(image)
        tags = _detect_apriltags(gray, self.tag_family)
        return _tags_to_detections(
            tags,
            width,
            height,
            confidence_threshold_pct=self.confidence_threshold_pct,
            bbox_padding_px=self.bbox_padding_px,
        )

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        camera = self._camera_for_request(camera_name)
        props = await camera.get_properties(timeout=timeout)
        intr = props.intrinsic_parameters
        if intr is None:
            raise Exception("camera must provide intrinsic parameters for 3D segmentation")
        intrinsics = (
            intr.focal_x_px,
            intr.focal_y_px,
            intr.center_x_px,
            intr.center_y_px,
        )

        cam_images, _ = await camera.get_images(timeout=timeout)
        source = _color_image_from_camera_images(cam_images)
        gray, width, height = _gray_from_viam_image(source)
        tags = _detect_apriltags(gray, self.tag_family)
        detections = _tags_to_detections(
            tags,
            width,
            height,
            confidence_threshold_pct=self.confidence_threshold_pct,
            bbox_padding_px=self.bbox_padding_px,
        )

        pcd_bytes, _ = await camera.get_point_cloud(timeout=timeout)
        all_points = read_pcd_xyz(pcd_bytes)

        objects: List[PointCloudObject] = []
        for det in detections:
            segment = filter_points_in_bbox(
                all_points,
                intrinsics,
                int(det.x_min),
                int(det.y_min),
                int(det.x_max),
                int(det.y_max),
            )
            if segment.shape[0] < self.min_segment_points:
                continue
            objects.append(
                PointCloudObject(
                    point_cloud=write_pcd_xyz(segment),
                    geometries=GeometriesInFrame(
                        reference_frame=camera_name,
                        geometries=[segment_geometry(segment, det.class_name)],
                    ),
                )
            )
        return objects

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> CaptureAllResult:
        if return_classifications:
            raise NotImplementedError()

        detections = None
        image = None
        objects = None
        if return_detections:
            detections = await self.get_detections_from_camera(
                camera_name, extra=extra, timeout=timeout
            )
        if return_object_point_clouds:
            objects = await self.get_object_point_clouds(
                camera_name, extra=extra, timeout=timeout
            )
        if return_image:
            camera = self._camera_for_request(camera_name)
            cam_images, _ = await camera.get_images(timeout=timeout)
            source = _color_image_from_camera_images(cam_images)
            image = ViamImage(source.data, source.mime_type)
        return CaptureAllResult(image=image, detections=detections, objects=objects)

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Any]:
        raise NotImplementedError()

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Any]:
        raise NotImplementedError()


async def run_module():
    module = ApriltagModule.from_args()
    for key in Registry.REGISTERED_RESOURCE_CREATORS().keys():
        module.add_model_from_registry(*key.split("/"))  # pyright: ignore [reportArgumentType]
    await module.start()


if __name__ == "__main__":
    asyncio.run(run_module())
