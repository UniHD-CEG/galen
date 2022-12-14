import ast
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from runtime.compress.torch_compress.torch_adapters import TorchCompressAdapter
from runtime.controller.algorithmic_mode import AlgorithmicMode
from runtime.data.data_provider import CIFAR10Provider
from runtime.evaluation.torch_evaluator import TorchOnlyEvaluator
from runtime.model.torch_model import TorchModelFactory
from runtime.sensitivity.sensitivity_analysis import SensitivityAnalysis
from tools.util.model_provider import load_model


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
        "--store_dir",
        type=str,
        default="./results/"
    )

    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="Path to the directory containing the data set.",
        type=str,
        default="./data/",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num_worker",
        type=int,
        default=4
    )

    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--normalization_constants",
        type=str,
        default="([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])"
    )

    parser.add_argument(
        "--enabled_methods",
        type=str,
        default='(["p-conv", "p-lin", "q-fp32", "q-int8", "q-mixed"])'
    )

    parser.add_argument(
        "--frozen_layers",
        type=str,
        default="{'p-lin': ['fc']}"
    )

    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=10
    )

    parser.add_argument(
        "--identifier",
        type=str,
        default=""
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=10
    )

    args, _ = parser.parse_known_args()
    return args


def construct_for_model(model, args):
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    data_provider = None
    if args.dataset == "cifar":
        data_provider = CIFAR10Provider(device,
                                        seed=args.seed,
                                        batch_size=args.batch_size,
                                        data_dir=args.data_dir,
                                        num_workers=args.num_worker,
                                        split_ratio=args.split_ratio,
                                        image_normalization_constants=ast.literal_eval(args.normalization_constants))
    model_factory = TorchModelFactory(model,
                                      batch_input_shape=data_provider.batch_input_shape,
                                      target_device=device,
                                      frozen_layers=ast.literal_eval(args.frozen_layers))
    reference_model = model_factory.get_reference_model()
    compress_adapter = TorchCompressAdapter(reference_model,
                                            model_factory=model_factory,
                                            enabled_methods=ast.literal_eval(args.enabled_methods),
                                            channel_round_to=1)
    model_evaluator = TorchOnlyEvaluator(data_provider,
                                         logging_service=None,
                                         target_device=device)
    sensitivity_analysis = SensitivityAnalysis(compress_adapter,
                                               model_factory=model_factory,
                                               model_evaluator=model_evaluator,
                                               alg_mode=AlgorithmicMode.INDEPENDENT,
                                               sampling_steps=args.sampling_steps)
    return sensitivity_analysis, compress_adapter, reference_model


def sensitivity_standalone(args):
    model, _ = load_model(args.model, num_classes=args.num_classes, checkpoint_path=args.ckpt_load_path)
    sens_analysis, compress_adapter, model_ref = construct_for_model(model, args)
    reference_policy = compress_adapter.get_reference_policy()
    return sens_analysis.analyse(model_ref, reference_policy)


if __name__ == '__main__':
    args = parse_arguments()
    results = sensitivity_standalone(args)

    target_dir = Path(args.store_dir)
    target_dir.mkdir(exist_ok=True, parents=True)
    target_file = target_dir / f"sens_results_{args.identifier}.pickle"
    with open(target_file, "wb") as file_handle:
        pickle.dump(results.layer_results, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
