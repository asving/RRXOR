#!/bin/bash
GPU=$1; shift
for cfg in "$@"; do
  IFS=: read arm m1 m2 seed cross <<< "$cfg"
  extra=""; [ -n "$cross" ] && [ "$cross" != "0" ] && extra="--cross $cross"
  CUDA_VISIBLE_DEVICES=$GPU ../.venv/bin/python train_route2.py --arm $arm --m1 $m1 --m2 $m2 --seed $seed --steps 4000 $extra
done
echo QUEUE2_DONE_GPU$GPU
