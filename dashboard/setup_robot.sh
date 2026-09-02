#!/bin/bash
# Fetch the DeepCybo Lite robot model for the dashboard's 3D viewer.
# Assembles dashboard/robot/ (URDF + STL meshes, ~170 MB — kept out of git)
# from the TeamLite-DeepCybo/lite_urdf repository.
set -euo pipefail
cd "$(dirname "$0")"
if [ -d robot/meshes ]; then
  echo "robot/ already set up"; exit 0
fi
tmp=$(mktemp -d)
git clone -q --depth 1 https://github.com/TeamLite-DeepCybo/lite_urdf.git "$tmp/lite_urdf"
mkdir -p robot
cp "$tmp/lite_urdf/urdf/lite_flash_arm_gripper.urdf" robot/
cp -r "$tmp/lite_urdf/meshes" robot/meshes
rm -rf "$tmp"
echo "robot/ ready ($(ls robot/meshes | wc -l | tr -d ' ') meshes)"
