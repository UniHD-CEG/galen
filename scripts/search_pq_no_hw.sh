#!/usr/bin/env bash

python -m tools.search_policy \
  --model resnet18_cifar \
  --ckpt_load_path results/checkpoints/cifar/resnet18_cifar.pth \
  --log_dir ./logs \
  --agent pruning-quantization-agent \
  --episodes 410 \
  --add_search_identifier exp_01c_pq \
  --alg_config num_workers=6 reward=r6 mixed_reference_bits=6 reward_target_cost_ratio=0.25 enable_latency_eval=False reward_episode_cost_key=BOPs