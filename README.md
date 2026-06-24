# `apriltag` module

A Viam module that uses apriltags as an implementation for a PoseTracker component.

Fork of [viam-labs/apriltag](https://github.com/viam-labs/apriltag) with fixes for module reconfigure when unrelated machine resources are disabled.

**Module:** [marcus-org/apriltag](https://app.viam.com/module/marcus-org/apriltag)  
**Model:** `marcus-org:apriltag:pose_tracker`

## Configuration and Usage

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/build/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in [the Viam app](https://app.viam.com/).
[Add a pose tracker component](https://docs.viam.com/build/configure/#components) using model **`marcus-org:apriltag:pose_tracker`**.

On the new component panel, copy and paste the following attribute template into your component's attributes field, editing the attributes as applicable:

```json
{
  "camera_name": "camera-1",
  "tag_family": "tag36h11",
  "tag_width_mm": 29.5
}
```

> [!NOTE]  
> For more information, see [Configure a Machine](https://docs.viam.com/manage/configuration/).

### Attributes

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `camera_name` | string | **Required** | The name of the camera to depend on. |
| `tag_family` | string | **Required** | The Apriltag 'tag family' to detect. |
| `tag_width_mm` | float | **Required** | The width of the tags to be detected, measured from tag corner to tag corner and specified in mm. |

### Generating Apriltags

To get quickly started tracking poses of Apriltags the example file `tag36h11_1-30.pdf` can be printed and used with the example configuration above. There exist a number of [online generators](https://shiqiliu-67.github.io/apriltag-generator/) that can be used to create similar files suitable to your specific needs.

For more information about the Apriltag specification and how they can be used see the [AprilRobotics repo](https://github.com/aprilrobotics/apriltag).
