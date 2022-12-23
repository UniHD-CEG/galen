#!/usr/bin/env bash

python -m tools.search_policy \
  --model resnet18_cifar \
  --ckpt_load_path results/checkpoints/cifar/resnet18_cifar.pth \
  --log_dir ./logs \
  --agent independent-single-layer-quantization \
  --episodes 310 \
  --add_search_identifier exp_01b_quantize \
  --alg_config num_workers=6 reward=r6 mixed_reference_bits=6 reward_target_cost_ratio=0.25 tvm_target='tvm.target.arm_cpu("rasp4b")' tvm_device_key=pi-cluster-head-pi4b tvm_rpc_ip=localhost tvm_rpc_port=9000
