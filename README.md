# `apriltag` module

A Viam module that uses apriltags as an implementation for a PoseTracker component, an annotated camera, and a 2D vision detector.

Fork of [viam-labs/apriltag](https://github.com/viam-labs/apriltag) with fixes for module reconfigure when unrelated machine resources are disabled.

**Module:** [marcus-org/apriltag](https://app.viam.com/module/marcus-org/apriltag)

## Models

| Model | API | Purpose |
| ----- | --- | ------- |
| `marcus-org:apriltag:pose_tracker` | pose tracker | 6-DOF tag poses via PnP |
| `marcus-org:apriltag:camera` | camera | Annotated debug feed |
| `marcus-org:apriltag:vision` | vision | 2D tag detections for `detections-to-segments` |

## Configuration and Usage

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/build/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in [the Viam app](https://app.viam.com/).

### Pose tracker

[Add a pose tracker component](https://docs.viam.com/build/configure/#components) using model **`marcus-org:apriltag:pose_tracker`**.

```json
{
    "camera_name": "crop-camera",
  "tag_family": "tag36h11",
  "tag_width_mm": 29.5
}
```

### Vision detector + detections-to-segments

Use **`marcus-org:apriltag:vision`** as the 2D detector, then wire it into **`viam:vision:detections-to-segments`** for 3D.

The detector returns **tight boxes** in the Control tab (`GetDetectionsFromCamera`), but **padded boxes** on `GetDetections` — the path `detections-to-segments` uses. Set `bbox_padding_px` to grow each tag bbox before 3D projection.

**1. Detector** (`marcus-org:apriltag:vision`):

```json
{
  "name": "apriltag-detector",
  "api": "rdk:service:vision",
  "model": "marcus-org:apriltag:vision",
  "attributes": {
    "camera_name": "crop-camera",
    "tag_family": "tag36h11",
    "confidence_threshold_pct": 0.0,
    "bbox_padding_px": 40
  }
}
```

**2. Segmenter** (`viam:vision:detections-to-segments`):

```json
{
  "name": "apriltag-segment",
  "api": "rdk:service:vision",
  "model": "viam:vision:detections-to-segments",
  "attributes": {
    "detector_name": "apriltag-detector",
    "camera_name": "crop-camera",
    "confidence_threshold_pct": 0.1,
    "mean_k": 1,
    "sigma": 2.5,
    "infer_minimum_depth": true
  }
}
```

```python
vision = VisionClient.from_robot(robot, "apriltag-segment")
objects = await vision.get_object_point_clouds("crop-camera")
```

<details>
<summary>Alternative: native 3D on the detector (skip detections-to-segments)</summary>

The same vision model also implements `GetObjectPointClouds` directly. Use this if `detections-to-segments` still returns empty segments.

</details>

Each object's `geometries.geometries[0].label` is the tag ID string; `center` is the 3D point in the camera frame.

> [!NOTE]  
> For more information, see [Configure a Machine](https://docs.viam.com/manage/configuration/).

### Attributes

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `camera_name` | string | **Required** | The name of the camera to depend on. |
| `tag_family` | string | **Required** | The Apriltag 'tag family' to detect. |
| `tag_width_mm` | float | Required for pose tracker / camera | Tag width in mm (corner to corner). Not used by the vision detector. |
| `confidence_threshold_pct` | float | Optional (vision only) | Detections below this confidence are dropped. Range `0.0`–`1.0`. Default `0.0`. Confidence is `decision_margin / 40`, capped at `1.0`. |
| `bbox_padding_px` | int | Optional (vision only) | Pixels to expand each bbox on **`GetDetections`** (the path used by `detections-to-segments`). Default `40`. `GetDetectionsFromCamera` always returns tight tag boxes. |
| `min_segment_points` | int | Optional (vision only) | Minimum depth points required to emit a 3D segment. Default `3`. |

> [!NOTE]
> `crop-camera` (or whichever depth camera you use) must support point clouds (`supports_pcd: true`).

### Generating Apriltags

To get quickly started tracking poses of Apriltags the example file `tag36h11_1-30.pdf` can be printed and used with the example configuration above. There exist a number of [online generators](https://shiqiliu-67.github.io/apriltag-generator/) that can be used to create similar files suitable to your specific needs.

For more information about the Apriltag specification and how they can be used see the [AprilRobotics repo](https://github.com/aprilrobotics/apriltag).

## Publishing a new module version

From the module root, with the [Viam CLI](https://docs.viam.com/cli/) authenticated:

```bash
# 1. Build the upload archive (linux/arm64 + linux/amd64 native libs)
make module.tar.gz

# 2. Upload for each platform your meta.json declares
viam module upload --version 0.2.3 --platform linux/amd64 --upload module.tar.gz
viam module upload --version 0.2.3 --platform linux/arm64 --upload module.tar.gz

# 3. Optional: refresh module metadata on the registry
viam module update --module ./meta.json
```

Bump `--version` to a new semver for each release. After upload, update the module version on your machine's **CONFIGURE** tab to pick up the new release.
