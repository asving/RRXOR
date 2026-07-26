#!/bin/bash
# usage: route_queue.sh <gpu> <configs...>   config = arm:m1:m2:seed[:ctl]
GPU=$1; shift
for cfg in "$@"; do
  IFS=: read arm m1 m2 seed ctl <<< "$cfg"
  extra=""; [ "$ctl" = "ctl" ] && extra="--noshared"
  CUDA_VISIBLE_DEVICES=$GPU ../.venv/bin/python train_route.py --arm $arm --m1 $m1 --m2 $m2 --seed $seed --steps 4000 $extra
done
echo QUEUE_DONE_GPU$GPU
