import ast
import os
from argparse import Namespace, ArgumentParser
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm


def parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--source_path",
        help="Source path of ImageNet Dataset",
        type=str,
        default="/local/datasets/imagenet"
    )

    parser.add_argument(
        "--destination_path",
        help="Destination path of the ImageNet Subset",
        type=str,
        default="/local/datasets/imagenet100"
    )

    parser.add_argument(
        "--class_file",
        help="File specifying the ImageNet classes to use",
        type=str,
        default="other/imagenet100.txt"
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="['train', 'val']"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=16
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    src_dir = Path(args.source_path)
    dst_dir = Path(args.destination_path)
    txt_path = Path(args.class_file)

    dst_dir.mkdir(parents=True, exist_ok=True)


    def copy_func(pair):
        src, dst = pair
        os.system(f'ln -s {src} {dst}')


    def empty_dir(pair):
        cls_name, root_path = pair
        path = root_path / cls_name
        path.mkdir(parents=True, exist_ok=True)


    splits = ast.literal_eval(args.splits)

    for split in splits:
        src_split_dir = src_dir / split
        dst_split_dir = dst_dir / split
        dst_split_dir.mkdir(parents=True, exist_ok=True)
        cls_list = []
        f = open(txt_path, 'r')
        for x in f:
            cls_list.append(x[:9])
        pair_list = [(src_split_dir / c, dst_split_dir / c) for c in cls_list]

        p = Pool(args.threads)

        for _ in tqdm(p.imap_unordered(copy_func, pair_list), total=len(pair_list)):
            pass

        # create empty dirs for all other classes -> consistent class numbers -> use retrained models w/o own classifier
        all_classes_pairs = [(entry.name, dst_split_dir) for entry in os.scandir(src_split_dir) if entry.is_dir()]
        for _ in tqdm(p.imap_unordered(empty_dir, all_classes_pairs), total=len(all_classes_pairs)):
            pass

        p.close()
        p.join()
