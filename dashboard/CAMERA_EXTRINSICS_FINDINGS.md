# DeepCybo Lite — Camera Extrinsics & Intrinsics Findings

**Date:** 2026-08-25
**Goal:** Authoritative HEAD/TOP (and wrist) camera mounting transform + intrinsics (FOV/focal/resolution) for the dashboard overlay.
**TL;DR:** The URDF `head_camera_joint` values **are authoritative and current** (byte-identical to the maintained `lite_urdf` source repo). `head_camera_link` is a **ROS optical frame** and the dashboard's `OPT2THREE` flip already handles orientation correctly — so the overlay misalignment is **almost entirely a wrong FOV**, not a wrong pose. **Use vertical FOV = 80°** for the head cam (not 58°).

---

## 1. HEAD / TOP CAMERA

### 1a. Extrinsic (mounting transform) — AUTHORITATIVE

| Field | Value |
|---|---|
| Joint | `head_camera_joint` (type=fixed) |
| Parent / reference frame | `world_root` |
| Child | `head_camera_link` |
| Translation xyz (m) | `(-1.937121605e-06, -0.1082753717, 1.010000795)` ≈ **(0, −0.1083, 1.0100)** |
| Rotation rpy (rad) | `(-2.3561871804, 0, 3.1415926264)` |
| Rotation rpy (deg) | **(−135.000°, 0.000°, +180.000°)** |
| Quaternion (x, y, z, w) | **(0, −0.923878, +0.382687, 0)** |
| Quaternion (w, x, y, z) | (0, 0, −0.923878, +0.382687) |

**Source (three independent, all identical):**
- `~/Documents/dreamzero/dashboard/robot/lite_flash_arm_gripper.urdf:688-689` (dashboard copy)
- `~/Documents/dreamzero/lite_urdf/urdf/lite_flash_arm_gripper.urdf:688-689` (authoritative maintained repo `TeamLite-DeepCybo/lite_urdf`)
- MJCF cross-check `TeamLite-DeepCybo/dated_lite_moveit2:lite_urdf/mjcf/lite.xml` — `<site name="head_camera_link" pos="-1.937121605e-06 -0.1082753717 1.010000795" quat="5.2025876e-09 -1.25600277e-08 -0.9238781338 0.382686809"/>` (quat wxyz matches to 6 dp).

These many-decimal values are CAD/URDF-export derived and are the current maintained mount.

### 1b. Optical-frame convention

`head_camera_link` is a **ROS optical frame** (REP-104: **+Z forward, +X right, +Y down**), tilted ~45° downward toward the workspace.

Derivation — rotation matrix R for rpy=(−135°,0,180°) gives the link axes expressed in `world_root`:
```
X_cam(right)   = [-1,      0,       0     ]   → world −X (image-right = robot-left)
Y_cam(down)    = [ 0,  +0.7071, -0.7071  ]   → down & +Y
Z_cam(forward) = [ 0,  -0.7071, -0.7071  ]   → into the scene, tilted 45° down  ✓
```
Z_cam points into the workspace and downward → this is the **optical forward (+Z)** axis, confirming the optical convention. The −135° roll = −90° (body→optical) plus an extra −45° downward pitch of the head. There is **no separate `head_camera_optical_joint`** in the `*_flash_arm_gripper` URDF variant (it's referenced only in `lite_urdf/mappings/name_mapping.json` for other variants); `head_camera_link` itself is the optical frame the dashboard uses.

### 1c. Intrinsics — AUTHORITATIVE per-unit calibration (fisheye)

Real ChArUco calibration of the physical head camera:

| Field | Value |
|---|---|
| Resolution | **1280 × 720** |
| fx, fy | **497.792527, 497.773123** |
| cx, cy | **650.230690, 337.600553** |
| Distortion model | **equidistant** (fisheye / Kannala-Brandt, OpenCV `fisheye`) |
| Distortion coeffs (k1..k4) | **[−0.027929, 0.012395, −0.010687, 0.002635]** |
| Derived HFOV (equidistant, θ=r/f) | **≈ 147°** (74.8° left + 72.5° right about cx) |
| Derived VFOV (equidistant) | **≈ 83°** (38.9° top + 44.0° bottom about cy) |
| Pinhole-equivalent HFOV / VFOV (ignoring fisheye) | 104.3° / 71.8° |

**Source:** `TeamLite-DeepCybo/lite_aruco_umi_ros2:config/camera_info_head.yaml` (produced by `scripts/calibrate_intrinsics.py --fisheye` from a ChArUco clip; raw `config/calib/head.npz` also present). Confirmed the real head lens is a **wide-angle fisheye (~147° H)**, *not* a rectilinear D435.

> ⚠️ **Resolution mismatch:** this calibration is 1280×720 (16:9), but the LeRobot training/eval dataset and the dashboard footage are **640×480 (4:3)** — a different sensor mode. The 1280×720 K matrix therefore cannot be scaled 1:1 to 640×480. For the dashboard, use the sim-matched FOV in §1d instead.

### 1d. Sim-matched FOV for 640×480 (what the dashboard should use)

RoboTwin renders the Lite head cam to match the real 640×480 footage:

| Field | Value |
|---|---|
| Profile `Lite_Real_Head` | **fovy = 80° (vertical), 640×480, near 0.05** |
| Implied HFOV (pinhole, 4:3) | 2·atan(tan40°·4/3) ≈ **96.4°** |
| Implied fy (px) | 240 / tan(40°) ≈ **286.0** |

**Source:** `~/Documents/dreamzero/Lite-Robotwin-2.0/task_config/_camera_config.yml:22-26`; README statement `Lite-Robotwin-2.0/README.md:123` — *"`Lite_Real_Head` at 640x480/80° vertical FOV"*; tuning script `script/calibrate_lite_camera_fov.py` (compares real episode frames vs pinhole fovy candidates; explicitly notes real views have barrel distortion that pinhole fovy matches only at central scale). Applied in sim at `envs/camera/camera.py:116` (`fovy=np.deg2rad(camera_config["fovy"])`).

**Cross-check:** calibration VFOV ≈ 83° (720p fisheye) vs RoboTwin pinhole fovy = 80° (480p) — agree within ~3°. **High confidence the effective vertical FOV ≈ 80°.** The previously eyeballed 58° is ~22° too narrow → this is the overlay-misalignment cause.

### 1e. Dataset confirmation of resolution
`A6000:${HOME}/dreamzero_eval/datasets/deepcybo_lite_bilateral_gear/meta/info.json` — `observation.images.image_head` shape `[480, 640, 3]`, h264, 30 fps. (Wrist streams identical 480×640.) No intrinsics stored in the LeRobot meta.

---

## 2. WRIST CAMERAS

### 2a. Extrinsics — AUTHORITATIVE (`lite_urdf`)

**Left** — `left_wrist_camera_joint`, parent `left_wrist_pitch_link`:
- xyz = `(0.1237537251, -0.1140010331, 0.02440096001)` m
- rpy = `(-2.3561994776, 0.0698124228, -2.2965665346)` rad = **(−135.001°, +4.000°, −131.583°)**

**Right** — `right_wrist_camera_joint`, parent `right_wrist_pitch_link`:
- xyz = `(0.1542184407, -0.06728357609, 0.02433415705)` m
- rpy = `(-2.3562047848, 0.0698124228, -1.9643445346)` rad = **(−135.001°, +4.000°, −112.549°)**

Both are ROS **optical** frames (same −135° roll pattern). Source: `lite_urdf/urdf/lite_flash_arm_gripper.urdf:691-699` (identical in dashboard copy).

### 2b. Wrist intrinsics — fallback (no per-unit calibration found)

RoboTwin `Lite_Real_Wrist`: **fovy = 125° (vertical), 640×480, near 0.005, forward_offset 0.008 m**, `frame_rotation_wxyz=[0.5,0.5,-0.5,0.5]` (that quat is a *SAPIEN* optical→sim axis convention — **not** needed for the dashboard, which uses `OPT2THREE`).
Source: `Lite-Robotwin-2.0/task_config/_camera_config.yml:28-35`; README:124. Very wide fisheye. No wrist ChArUco calibration file exists (only `head.npz`), so 125° V is the best available number.

---

## 3. CAMERA HARDWARE MODEL

- **Physical head + wrist cams:** generic **wide-angle fisheye USB (UVC) cameras**, driven by `usb_cam` (MJPEG passthrough, `YUV422P`, `/dev/videoN`) — see `lite_vision_ros2/src/usb_cam-main/config/params_1.yaml`. The `equidistant` (fisheye) calibration model confirms fisheye optics, ~147°H/83°V for the head. Exact module not named in any repo.
- **"D435" / "L515" strings are NOT the physical device.** They appear only as (a) RoboTwin **sim** camera presets (`_camera_config.yml` `D435: fovy 37`, `L515: fovy 45`) and (b) a web-platform UI dropdown default (`web-platform:components/cloud-evaluation-console.tsx` `cameraOptions=["D435","L515","RealSense","None"]`). No RealSense (`realsense2_camera`) or Orbbec driver exists anywhere in the org. The RoboTwin embodiment `config.yml:123-128` head_camera `type: D435` is a nominal sim placeholder.
- ⚠️ Because the head cam is fisheye, no rectilinear "factory-default" intrinsic applies; the calibrated fisheye K in §1c is the only per-unit truth (at 1280×720).

---

## 4. VERDICT

- **Are the URDF `head_camera_joint` values authoritative and current?** **YES.** They are byte-identical to the maintained `TeamLite-DeepCybo/lite_urdf` source and cross-check against the MoveIt MJCF quaternion. No better extrinsic source exists.
- **Is the extrinsic already handled correctly in the dashboard?** **YES.** `app.js:376-381` takes the `head_camera_link` world pose from URDF FK and multiplies by `OPT2THREE` (180° about X, `app.js:341`) — the exact ROS-optical→three.js(OpenGL) conversion. Placement + orientation are correct; **no positional/rotational offset is needed.**
- **What's actually wrong:** the **FOV**. Default camera fov is 50° (`app.js:26`) / slider-driven, and the team eyeballed ~58°. The real effective vertical FOV is **≈ 80°**.

---

## 5. RECOMMENDATION FOR THE DASHBOARD (`data/profiles.json`)

Keep the head cam **link-mounted** (do NOT switch to a fixed `cameraPose` — link mode preserves the correct optical roll via the full quaternion; a `lookAt`-with-up=(0,1,0) pose would introduce a ~45° roll error). Only fix the FOV.

For the `deepcybo` head camera entry, add a default calibration whose **orientation/position offsets are zero** (URDF + `OPT2THREE` already correct) and whose **fov = 80**:

```jsonc
// profiles.json → deepcybo.cameras[ id:"head" ]
{
  "id": "head",
  "label": "head cam",
  "link": "head_camera_link",
  "framesBase": "data/dashboard_data/frames_seq",
  "calib": { "dp": [0, 0, 0], "eul": [0, 0, 0], "fov": 80 }   // vertical FOV, deg
}
```

Wrist cams (same pattern):
```jsonc
{ "id": "wl", ..., "calib": { "dp":[0,0,0], "eul":[0,0,0], "fov": 125 } }
{ "id": "wr", ..., "calib": { "dp":[0,0,0], "eul":[0,0,0], "fov": 125 } }
```

Notes / small code touch-ups:
- `three.PerspectiveCamera.fov` is the **vertical** FOV — 80 maps directly. Keep `aspect = IMG_ASPECT = 4/3` (`app.js:37`), which matches 640×480. ✓
- The dashboard currently only loads `calib` from `localStorage` (`app.js:348 loadCalib()`); to honor a profiles.json default, fall back to `camDef().calib` when localStorage is empty. Alternatively (quick fix) just set the `#fov` slider default to **80** in `index.html`.
- The auto-calibrator clamps solved fov to `[30, 90]` (`app.js:586`) — 80 is fine, but raise the upper clamp to ≥130 if you ever solve the **wrist** cams (125°) through that tool. Setting `calib.fov` directly (as above) bypasses the clamp (`app.js:403`).
- Fine-tuning: if a residual scale mismatch remains, sweep fov in 78–83° (the calibration/sim agreement band) rather than going back toward 58°.

---

## 6. PROVENANCE INDEX (every number traceable)

| Value | Source |
|---|---|
| Head extrinsic xyz/rpy | `lite_urdf/urdf/lite_flash_arm_gripper.urdf:688-689` (+ dashboard copy, + `dated_lite_moveit2` mjcf quat) |
| Head fx/fy/cx/cy, fisheye distortion, 1280×720 | `TeamLite-DeepCybo/lite_aruco_umi_ros2:config/camera_info_head.yaml` |
| Head FOV 80° @640×480 | `Lite-Robotwin-2.0/task_config/_camera_config.yml:22-26`; `Lite-Robotwin-2.0/README.md:123`; `envs/camera/camera.py:116` |
| Dataset res 640×480 | `A6000:.../deepcybo_lite_bilateral_gear/meta/info.json` (`observation.images.image_head` shape [480,640,3]) |
| Wrist extrinsics | `lite_urdf/urdf/lite_flash_arm_gripper.urdf:691-699` |
| Wrist FOV 125° | `Lite-Robotwin-2.0/task_config/_camera_config.yml:28-35`; `README.md:124` |
| Hardware = USB fisheye (usb_cam) | `lite_vision_ros2/src/usb_cam-main/config/params_1.yaml`, `config/camera_info.yaml` (README: placeholder), distortion_model=`equidistant` |
| "D435" is sim/UI placeholder only | `_camera_config.yml:12-19`; `Lite-Robotwin-Embodiment/config.yml:123-128`; `web-platform:components/cloud-evaluation-console.tsx` |
| Dashboard optical→GL handling | `dashboard/app.js:341` (OPT2THREE), `:376-410` (camBasePose/syncCamView) |
