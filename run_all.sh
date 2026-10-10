#!/bin/bash
# Runs every step of the film in order. Measured: 3 min 27 s on a MacBook Pro M4 (all steps, no GPU).
set -e
cd "$(dirname "$0")"
python3 tests.py
for s in step1_task step2_deep step3_batch_norm step4_the_reason step5_the_test step6_smoother step7_the_cost step8_layer_norm exercise; do
  echo
  echo "== $s"
  python3 $s.py
done
