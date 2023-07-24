# Galen: Hardware-specific Automatic Compression of Neural Networks

Galen is a framework for automatic compression of neural networks by applying layer-specific pruning and quantization.
The layer-wise compression is determined by a RL algorithm which uses the sensitivity but also hardware inference latency as features.

We presented the details of the algorithm at the Practical-DL workshop collocated with AAAI-23:
[**Towards Hardware-Specific Automatic Compression of Neural Networks**
](https://arxiv.org/abs/2212.07818) ([AAAI-23: 2nd International Workshop on Practical
Deep Learning in the Wild](https://practical-dl.github.io/))

Happy to announce that we received the best paper award of this workshop!

**Abstract**:

_Compressing neural network architectures is important to allow the deployment of models to embedded or mobile devices,
and pruning and quantization are the major approaches to compress neural networks nowadays. Both methods benefit when
compression parameters are selected specifically for each layer. Finding good combinations of compression parameters,
so-called compression policies, is hard as the problem spans an exponentially large search space. Effective compression
policies consider the influence of the specific hardware architecture on the used compression methods. We propose an
algorithmic framework called Galen to search such policies using reinforcement learning utilizing pruning and
quantization, thus providing automatic compression for neural networks. Contrary to other approaches we use inference
latency measured on the target hardware device as an optimization goal. With that, the framework supports the
compression of models specific to a given hardware target. We validate our approach using three different reinforcement
learning agents for pruning, quantization and joint pruning and quantization. Besides proving the functionality of our
approach we were able to compress a ResNet18 for CIFAR-10, on an embedded ARM processor, to 20% of the original
inference latency without significant loss of accuracy. Moreover, we can demonstrate that a joint search and compression
using pruning and quantization is superior to an individual search for policies using a single compression method._

![Algorithmic Schema](./figures/alg_schema.drawio.svg)

## Dependencies

Most important dependencies of the project are PyTorch and Apache TVM. To install all dependencies first create a new
conda environment using the included `environment.yml` by:

```shell
conda env create --name galen --file=environments.yml

# activate created environment
conda activate galen
```

### Apache TVM

It is recommended to build Apache TVM from source when you want to test with hardware feedback. For tests
without hardware feedback you could install using pip:

```shell
# CPU build only
pip install apache-tvm
```

#### Manual installation

To build from source using the created conda environment:

```shell
conda activate galen
cd ..
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build

cp ../cmake_build_config.cmake ./build/config.cmake
# or use the original tvm cmake config / configure by yourself (details: https://tvm.apache.org/docs/install/from_source.html#build-the-shared-library):
# cp cmake/config.cmake build

cd build
cmake ..

# -j specifies the number of compile threads
make -j4
```

To make the TVM python library usable on your system, add the following to your `.bashrc` (or `.zshrc` (...)):

```shell
export TVM_HOME=/path/to/tvm # replace with tvm repository location on your system
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

## Run a Search: No Hardware Feedback

### Pre-Train the ResNet18 on CIFAR-10

To search a compression policy for a ResNet18 model you first have to train the model on CIFAR-10. The model itself and
the used data is downloaded automatically. To train for 100 epochs, use the provided script:

```shell
bash ./scripts/train_resnet18.sh
```

### Search Policies

To run a compression search using the joint pruning & quantization agent without measuring the latency:

```shell
bash ./scripts/search_pq_no_hw.sh
```

Per episode a result dict will be stored as pickle file in the directory `./logs`. The result dict holds the search
accuracy per episode but also the found policy and additional metrics.

### Retrain & Test a Compressed Model

After searching for policies the compressed model has to be retrained first, before testing the achieved accuracy.
To retrain and test, select any episode result stored in the `./logs` directory. Adapt the path in `scripts/retrain.sh`
for the selected pickle file. Afterwards run by:

```shell
bash ./scripts/retrain.sh
```

This automatically compresses the model with the selected policy, retrains the model for 30 epochs and validates the
test accuracy using the CIFAR-10 test dataset.

## Run a Search: Measure Hardware Latency

To run a search with measured hardware latency some extra effort is required. First you have to set-up a TVM remote
device. Currently, the implementation supports ARM devices only.
For setting up a TVM remote device please refer to the TVM
documentation [[1]](https://tvm.apache.org/docs/tutorial/cross_compilation_and_rpc.html) (the section regarding
cross-compilation is not relevant, as this is done by the provided implementation).

**Adapt following parameters in subsequent scripts accordingly:**

- tvm_target
- tvm_device_key
- tvm_rpc_ip
- tvm_rpc_port

The pre-training and retraining steps are the same as for the search without latency measurement. To run a search using
the pruning agent:

```shell
bash ./scripts/search_p.sh
```

For a search using the quantization agent:

```shell
bash ./scripts/search_q.sh
```

Finally, using the joint agent:

```shell
bash ./scripts/search_pq.sh
```

To deactivate measurement of hardware latency, add `enable_latency_eval=False` to the `--alg_config` argument when using the scripts.

[1] https://tvm.apache.org/docs/tutorial/cross_compilation_and_rpc.html

### Apache-TVM: Missing Property `kernel.data.shape`

If you run into an issue that the `kernel` object has no property or attribute `data` or `data.shape` while running the
search (during the hardware latency measurement phase) you have to change a single source file of the Apache
TVM python package.

- navigate to the above cloned TVM repository on your machine
- open the file `python/tvm/topi/arm_cpu/bitserial_conv2d.py`
  - comment out the if statement in line 455 (`if len(kernel.data.shape) == 4:`)
  - fix indention for line 456 to 467

# Prune Pretrained Models

## Method 1 (preferred)

Define additional models at `tools/util/model_provider.py`. Provide a checkpoint file and reference it using the `--ckpt_load_path` argument when using scripts (see `scripts/search_p_custom_checkpoint.sh`).
The checkpoint is typically saved using `torch.save(model.state_dict(), PATH)` and contains only the network weights.

## Method 2

Provide a pretrained model and reference it using the `--model` argument when using scripts (see `scripts/search_p_custom_model.sh`).
The model is typically saved using `torch.save(model, PATH)` and contains the whole model.

## Retrain

Do not forget to retrain your model after pruning using the `scripts/retrain.sh` script. Specify the model using the `--model` argument, the checkpoint using the `--ckpt_load_path` argument and the pruning policy generated during the pruning search using the `--policy` argument.

## Reward function

Multiple reward functions can be specified using e.g the `reward=r6` argument for the R6 reward. For more details, take a look at the definitions at `runtime/agent/reward.py` and the script at `scripts/search_p.sh`.

Specify the `reward_target_cost_ratio=<n>` as a target for model-complexity. A value of 0.25 means, that the algorithm should reduce the model-complexity to 25% of the original model-complexity.

For the reward functions, the beta value can be defined using e.g. the`r6_beta=<n>` argument in case of the R6 reward. A beta value of -5 puts more emphasis on complexity reduction, a value of -1 puts more emphasis on accuracy.

## Dataset

Currently only cifar10 and imagenet are supported as datasets. More datasets can be added at `runtime/data/data_provider.py`.
