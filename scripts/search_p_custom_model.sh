#!/usr/bin/env bash

python -m tools.search_policy \
  --model ./pretrained_models/densenet121.pth \
  --log_dir ./logs/densenet121 \
  --agent independent-single-layer-pruning \
  --episodes 410 \
  --add_search_identifier densenet121 \
  --alg_config num_workers=6 reward=r6 r6_beta=-5 mixed_reference_bits=6 reward_target_cost_ratio=0.25 enable_latency_eval=False reward_episode_cost_key=BOPs
