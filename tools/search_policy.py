import os
from argparse import ArgumentParser, Namespace

import torch.cuda
import wandb

from runtime.agent.agent import GenericAgentConfig
from runtime.agent.pq_agent import PruningQuantizationAgent, PruningQuantizationAgentConfig
from runtime.agent.pruning_agent import IterativeSingleLayerPruningAgent, IndependentSingleLayerPruningAgent
from runtime.agent.quantization_agent import IterativeSingleLayerQuantizationAgent, QuantizationAgentConfig, \
    IndependentSingleLayerQuantizationAgent, Q3AIndependent
from runtime.torch_recipe import TorchConfiguration, TorchRecipe
from tools.util.model_provider import load_model

ENV_JOB_ID = "SLURM_JOB_ID"


def parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        help="Model for repo model use 'model@repo'",
        type=str,
        default="resnet18_cifar"
    )

    parser.add_argument(
        "--ckpt_load_path",
        dest="ckpt_load_path",
        help="Path to the model checkpoint data to load from. (To continue training from or test)",
        type=str,
        default=None,
        required=False
    )

    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="Path to the directory containing the data set.",
        type=str,
        default="./data/",
    )

    parser.add_argument(
        "--log_dir",
        dest="log_dir",
        help="Directory to save logs and results to",
        type=str,
        default="./logs"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None
    )

    parser.add_argument(
        "--add_search_identifier",
        type=str,
        default=""
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="dummy"
    )

    parser.add_argument(
        "--alg_config",
        metavar="KEY=VALUE",
        nargs="+",
        help="Specify algorithmic configuration parameters and hyperparameters in key=value format. "
             "Available parameters depend on selected agent."
    )

    return parser.parse_args()


def agent_provider(agent_identifier: str, device: torch.device, config: dict):
    if agent_identifier == "iterative-single-layer-pruning":
        return IterativeSingleLayerPruningAgent(device, agent_config=GenericAgentConfig(**config))
    if agent_identifier == "independent-single-layer-pruning":
        return IndependentSingleLayerPruningAgent(device, agent_config=GenericAgentConfig(**config))
    if agent_identifier == "iterative-single-layer-quantization":
        return IterativeSingleLayerQuantizationAgent(device, agent_config=QuantizationAgentConfig(**config))
    if agent_identifier == "independent-single-layer-quantization":
        return IndependentSingleLayerQuantizationAgent(device, agent_config=QuantizationAgentConfig(**config))
    if agent_identifier == "independent-3-action-quantization":
        return Q3AIndependent(device, agent_config=QuantizationAgentConfig(**config))
    if agent_identifier == "pruning-quantization-agent":
        return PruningQuantizationAgent(device, agent_config=PruningQuantizationAgentConfig(**config))
    raise Exception("Not a valid agent_identifier")


def fetch_env_config():
    env_config = dict()
    if ENV_JOB_ID in os.environ:
        env_config["JOB_ID"] = os.getenv(ENV_JOB_ID)
    return env_config


if __name__ == '__main__':
    args = parse_arguments()
    alg_config = dict(map(lambda s: s.split('='), args.alg_config))
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    agent = agent_provider(args.agent, device, alg_config)
    search_id = f"{agent.get_search_id()}_{args.add_search_identifier}"

    runtime_config = TorchConfiguration(target_device=device,
                                        log_dir=args.log_dir,
                                        data_dir=args.data_dir,
                                        enabled_methods=agent.supported_method(),
                                        search_identifier=search_id,
                                        **agent.config_overrides(),
                                        **alg_config)

    run_config = vars(runtime_config) | agent.config() | fetch_env_config()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=run_config, name=search_id)

    model, _ = load_model(args.model, num_classes=runtime_config.num_classes, checkpoint_path=args.ckpt_load_path)
    runtime_controller, original_model_handle = TorchRecipe(agent, runtime_config).construct_application(model)
    runtime_controller.search(args.episodes, original_model_handle)
