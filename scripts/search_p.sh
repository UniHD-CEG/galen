#!/usr/bin/env bash

python -m tools.search_policy \
  --model cifar10_resnet20@chenyaofo/pytorch-cifar-models \
  --log_dir ./logs \
  --agent independent-single-layer-pruning \
  --episodes 410 \
  --add_search_identifier exp_01a_prune \
  --alg_config num_workers=6 reward=r6 mixed_reference_bits=6 reward_target_cost_ratio=0.25 enable_latency_eval=False reward_episode_cost_key=BOPs
