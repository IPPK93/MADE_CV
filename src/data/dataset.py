from torchvision import transforms
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image
from pathlib import Path
import json


class SportDataset(Dataset):
    def __init__(
        self,
        mode: str = "train",
        transform: transforms = transforms.Resize(size=224),
    ):
        super().__init__()
        self.mode = mode
        df = pd.read_csv(f"data/raw/{self.mode}.csv")

        if self.mode == "train":
            self.labels = df["label"].to_list()
            with open("data/interim/label_to_id.json", "r") as file:
                self.label_to_id = json.load(file)
            with open("data/interim/id_to_label.json", "r") as file:
                self.id_to_label = json.load(file)

        self.image_ids = df["image_id"].to_list()
        self.p = Path(f"data/raw/{mode}")
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(str(self.p.joinpath(self.image_ids[idx]))).convert(
            "RGB"
        )

        image = self.transform(image)
        if self.mode == "train":
            return image, self.label_to_id[self.labels[idx]]
        return image
