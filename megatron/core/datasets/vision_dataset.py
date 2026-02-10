import os
import json
import hashlib
import numpy as np
import torch
from typing import Dict, Optional, Union, List, Tuple
from collections import OrderedDict
from abc import ABC
from PIL import Image
from megatron.core.datasets.megatron_dataset import MegatronDataset
from torchvision import transforms

class VisionLowLevelDataset:
    """Minimal low-level vision dataset.

    Each item is (image_path, label).
    """

    def __init__(self, samples: List[Tuple[str, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

import json
import hashlib
import numpy as np
import torch
from typing import Dict, Optional, Union, List
from collections import OrderedDict
from abc import ABC
from PIL import Image

class MegatronVisionDataset(MegatronDataset):
    """Megatron-compatible vision dataset.

    Returns:
        {
            "images": FloatTensor [C, H, W],
            "labels": LongTensor []
        }
    """

    def __init__(
        self,
        dataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split,
        config,
        transform=None,
    ):
        super().__init__(
            dataset=dataset,
            dataset_path=dataset_path,
            indices=indices,
            num_samples=num_samples,
            index_split=index_split,
            config=config,
        )
        print(f"\n\n\n {type(config.image_size)} \n\n\n")
        image_size = config.image_size
        self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),  # float32, [0,1]
            ])

    # ------------------------
    # Required static methods
    # ------------------------

    @staticmethod
    def numel_low_level_dataset(low_level_dataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config):
        """Builds a low-level dataset from a directory structure:

        dataset_path/
            class0/
                img1.jpg
            class1/
                img2.jpg
        """
        samples = []
        class_to_idx = {}  

        for class_idx, class_name in enumerate(sorted(os.listdir(dataset_path))):
            class_dir = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_to_idx[class_name] = class_idx
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif")):
                    samples.append(
                        (os.path.join(class_dir, fname), class_idx)
                    )

        return VisionLowLevelDataset(samples)

    @staticmethod
    def _key_config_attributes() -> List[str]:
        # Vision-specific knobs that should invalidate cache
        return [
            "random_seed",
            "image_size",
            "split",
            "split_matrix",
        ]

    # ------------------------
    # Dataset interface
    # ------------------------
    def __len__(self) -> int:
        if self.num_samples is not None:
            return self.num_samples
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        real_idx = self.indices[idx]
        image_path, label = self.dataset[real_idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(
                np.array(image)
            ).permute(2, 0, 1).float() / 255.0

        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.float32),
        }
