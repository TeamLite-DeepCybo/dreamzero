#!/bin/bash
# Pull every dense-grid eval run from the A6000 and rebuild dashboard data.
# Any results/<name>_dense/chunks.npz on the A6000 becomes a dashboard run —
# including runs pushed there from other machines (B300 etc.).
cd "$(dirname "$0")"
# Point DASH_REMOTE at the storage machine before running, e.g.:
#   export DASH_REMOTE=<user>@192.168.100.124   (or configure an ssh alias)
WAM="${DASH_REMOTE:-192.168.100.124}"
CP="-o ControlPath=$HOME/.ssh/sockets/wam"
for run in $(ssh $CP $WAM 'cd ~/dreamzero_eval/results && ls -d *_dense/chunks.npz 2>/dev/null | cut -d/ -f1'); do
  scp -q $CP $WAM:~/dreamzero_eval/results/$run/chunks.npz data/$run.npz && echo "$run synced"
done
.venv/bin/python build_data.py
