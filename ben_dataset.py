import os
import pickle
import random
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import h5py

BAND_MEAN = torch.tensor([
    429.9430203,              # B02
    614.21682446,             # B03
    590.23569706,             # B04
    -12.619993741972035,      # VV
    -19.29044597721542,       # VH
], dtype=torch.float32).view(5, 1, 1)

BAND_STD = torch.tensor([
    572.41639287,             # B02
    582.87945694,             # B03
    675.88746967,             # B04
    5.115911777546365,        # VV
    5.464428464912864,        # VH
], dtype=torch.float32).view(5, 1, 1)

class BEN10Dataset(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        inp_name=None,
        image_size=256,
        h5_path=None,
        max_samples=None,
    ):
        self.root = root
        self.split = split
        self.transform = transform

        # HDF5 文件路径
        self.image_size = int(image_size)
        self.h5_path = h5_path or os.path.join(
            root, f"ben_10p_clean_622_{self.image_size}.h5"
        )
        self.h5_file = None  # 延迟打开，防止多进程死锁
        self.h5_images = None
        self.h5_labels = None
        
        # 依然保留文本索引，用于获取图像的名字 (Name)
        self.index_file = os.path.join(
            root, f"processed_pt_{self.image_size}_clean622", f"{split}.txt"
        )

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(
                f"HDF5 file not found: {self.h5_path}. "
                f"Please check --image-size or pass explicit h5_path."
            )
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(
                f"Index file not found: {self.index_file}. "
                "Please ensure processed_pt_<image_size>_clean622 exists."
            )

        with open(self.index_file, "r", encoding="utf-8") as f:
            self.files = [line.strip() for line in f if line.strip()]

        # 如果需要截断数据 (Sanity Check)
        if max_samples is not None and max_samples > 0 and max_samples < len(self.files):
            # 注意：如果启用了截断，必须记录截断的索引，因为 HDF5 是按索引读取的
            rng = random.Random(3407)
            self.valid_indices = rng.sample(range(len(self.files)), max_samples)
            self.files = [self.files[i] for i in self.valid_indices]
        else:
            self.valid_indices = list(range(len(self.files)))

        self.num_classes = 19

        if not inp_name or not os.path.exists(inp_name):
            raise FileNotFoundError(
                f"Embedding file not found: {inp_name}. "
                "Please generate/check `bigearthnet19_glove_word2vec.pkl`."
            )

        with open(inp_name, "rb") as f:
            self.inp = torch.tensor(pickle.load(f), dtype=torch.float32)

        print(f"[BEN10Dataset-HDF5] {split}: {len(self.valid_indices)} samples loaded.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # 延迟打开 HDF5 文件（每个 DataLoader worker 都会独立打开一次）
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.h5_images = self.h5_file[f"{self.split}/images"]
            self.h5_labels = self.h5_file[f"{self.split}/labels"]

        # 获取在原始 HDF5 文件中真正的索引位置
        real_idx = self.valid_indices[index]

        # 瞬间连续读取大文件中的数组
        fusion_np = self.h5_images[real_idx]
        target_np = self.h5_labels[real_idx]

        fusion = torch.from_numpy(fusion_np)
        target = torch.from_numpy(target_np)
        name = self.files[index]

        # 归一化
        fusion = (fusion - BAND_MEAN) / BAND_STD
        
        # 数据增强：与 SAR-SLICO 版本保持一致的 4 种离散增强
        if self.split == "train":
            aug_type = random.choice([
            "orig",
            "hflip",
            "vflip",
            "rot180",
        ])
        else:
            aug_type = "orig"

        if aug_type == "hflip":
            fusion = TF.hflip(fusion)
        elif aug_type == "vflip":
            fusion = TF.vflip(fusion)
        elif aug_type == "rot180":
            fusion = torch.rot90(fusion, k=2, dims=[1, 2])

        return (fusion, name, [self.inp]), target
