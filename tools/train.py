import ast
from argparse import Namespace, ArgumentParser

import torch
import wandb

from runtime.data.data_provider import CIFAR10Provider
from runtime.data.imagenet_provider import ImageNetDataProvider
from tools.util import model_provider
from tools.util.compress import compress_model
from tools.util.model_provider import load_checkpoint
from tools.util.trainer import Trainer


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
        "--compressed_ckpt",
        dest="compressed_ckpt",
        help="Path to the model checkpoint data to load from. (To continue training from or test)",
        type=str,
        default=None,
        required=False
    )

    parser.add_argument(
        "--policy",
        help="Specify a compression policy / episode logging dict to compress the model with",
        default=None,
        required=False
    )

    parser.add_argument(
        "--store_dir",
        type=str,
        default="./results/checkpoints/cifar/"
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
        type=str,
        default="./logs/",
    )

    parser.add_argument(
        "--log_name",
        dest="log_name",
        type=str,
        default="train.out",
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
        "--add_train_identifier",
        type=str,
        default=""
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
        "--train_lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--train_mom",
        type=float,
        default=0.4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4
    )

    parser.add_argument(
        "--use_adadelta",
        action="store_true"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

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

    if args.dataset == "imagenet":
        data_provider = ImageNetDataProvider(device,
                                             data_dir=args.data_dir,
                                             batch_size=args.batch_size,
                                             seed=args.seed,
                                             num_workers=args.num_worker)

    model, model_name = model_provider.load_model(args.model, data_provider.num_classes, args.ckpt_load_path)

    print(model)

    epochs = args.epochs

    trainer = Trainer(
        data_provider,
        device,
        train_epochs=epochs,
        train_lr=args.train_lr,
        train_mom=args.train_mom,
        weight_decay=args.weight_decay,
        use_adadelta=args.use_adadelta,
        store_dir=args.store_dir,
        add_identifier=args.add_train_identifier,
        model_name=model_name,
        log_dir=args.log_dir,
        log_file_name=args.log_name
    )

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args),
               name=trainer.create_identifier(args.epochs))

    protocol = None
    if args.policy:
        enabled_methods = tuple(["p-conv", "p-lin", "q-fp32", "q-int8", "q-mixed"])
        model, protocol = compress_model(model, args.policy, data_provider, device, enabled_methods)
        checkpoint_path = args.compressed_ckpt
        if checkpoint_path is not None:
            model.load_state_dict(load_checkpoint(checkpoint_path))
        print(model)

    best_model = trainer.train(model=model)

    test_acc = trainer.test(best_model)
    trainer.store_logs({
        "compression_protocol": protocol
    })
    print(f"Acc. for test: {test_acc}")
