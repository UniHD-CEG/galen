#!/usr/bin/env bash

python -m tools.search_policy \
  --model resnet18_pretrained \
  --ckpt_load_path ./results/checkpoints/resnet18_pretrained/resnet18_pretrained_pre-train_lr0.05_mom0.9_ep93.pth \
  --log_dir ./logs/resnet18_pretrained \
  --agent independent-single-layer-pruning \
  --episodes 410 \
  --add_search_identifier resnet18_pretrained \
  --alg_config num_workers=6 reward=r6 r6_beta=-5 mixed_reference_bits=6 reward_target_cost_ratio=0.25 enable_latency_eval=False reward_episode_cost_key=BOPs
