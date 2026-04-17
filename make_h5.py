# python make_h5.py \
#   --root /media/sata/xyx/BigEarthNet/dataset \
#   --image-size 128 \
#   --splits train val

import argparse
import os

import h5py
import torch
from tqdm import tqdm

DEFAULT_ROOT = "/media/sata/xyx/BigEarthNet/dataset"
DEFAULT_IMAGE_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Pack processed pt files into HDF5.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="dataset root")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="image size")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="optional processed pt directory. default: <root>/processed_pt_<image_size>_clean622",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default="",
        help="optional output h5 path. default: <root>/ben_10p_clean_622_<image_size>.h5",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="splits to pack",
    )
    return parser.parse_args()


def infer_hw_from_sample(sample):
    image = sample["image"]
    if image.ndim != 3:
        raise ValueError(f"Invalid image shape: {tuple(image.shape)}")
    channels, height, width = image.shape
    if channels != 5:
        raise ValueError(f"Expected 5 channels but got {channels}")
    return int(height), int(width)


def create_h5(data_dir, h5_path, splits):
    with h5py.File(h5_path, "w") as h5f:
        for split in splits:
            txt_path = os.path.join(data_dir, f"{split}.txt")
            if not os.path.exists(txt_path):
                print(f"[Skip] Missing split index: {txt_path}")
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                files = [line.strip() for line in f if line.strip()]

            n_samples = len(files)
            if n_samples == 0:
                print(f"[Skip] Empty split: {split}")
                continue

            print(f"[{split}] packing {n_samples} samples...")
            first_pt_path = os.path.join(data_dir, split, files[0])
            first_sample = torch.load(first_pt_path, map_location="cpu", weights_only=True)
            height, width = infer_hw_from_sample(first_sample)

            img_ds = h5f.create_dataset(
                f"{split}/images",
                shape=(n_samples, 5, height, width),
                dtype="float32",
                chunks=(1, 5, height, width),
            )
            lbl_ds = h5f.create_dataset(f"{split}/labels", shape=(n_samples, 19), dtype="float32")

            for i, fname in enumerate(tqdm(files, desc=f"pack-{split}")):
                pt_path = os.path.join(data_dir, split, fname)
                sample = torch.load(pt_path, map_location="cpu", weights_only=True)
                image = sample["image"]
                label = sample["label"]

                if tuple(image.shape) != (5, height, width):
                    raise ValueError(
                        f"Inconsistent image shape in {pt_path}: {tuple(image.shape)} vs expected "
                        f"(5, {height}, {width})"
                    )

                img_ds[i] = image.numpy()
                lbl_ds[i] = label.numpy()


def main():
    args = parse_args()
    data_dir = args.data_dir or os.path.join(args.root, f"processed_pt_{args.image_size}_clean622")
    h5_path = args.h5_path or os.path.join(args.root, f"ben_10p_clean_622_{args.image_size}.h5")

    print(f"[Config] data_dir   : {data_dir}")
    print(f"[Config] h5_path    : {h5_path}")
    print(f"[Config] splits     : {args.splits}")

    create_h5(data_dir=data_dir, h5_path=h5_path, splits=args.splits)
    print(f"[Done] H5 saved to: {h5_path}")


if __name__ == "__main__":
    main()
