from src.data.dataset import SportDataset

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import EfficientNet_V2_M_Weights
import numpy as np

from typing import Union, Tuple, List


def make_dataloader(
    mode: str,
    transforms=EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms(),
    batch_size: int = 16,
    return_ids: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, List[str]]]:
    dataset = SportDataset(mode=mode, transform=transforms)

    if mode == "train":
        label_ids = [dataset.label_to_id[label] for label in dataset.labels]

        class_sample_count = np.array(
            [len(np.where(label_ids == t)[0]) for t in np.unique(label_ids)]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in label_ids])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler
        )
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if not return_ids:
        return dataloader
    return dataloader, dataset.image_ids
