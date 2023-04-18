from torchvision.models import EfficientNet
import torch

from tqdm import tqdm
import pandas as pd
import json

import logging

from src.data.make_dataloader import make_dataloader

logging.basicConfig(level=logging.INFO)


def predict_model(model_path: str, mode: str = "test") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading model...")

    model: EfficientNet = torch.load(model_path).to(device)
    model.eval()

    logging.info("Making dataloader...")

    test_dataloader, image_ids = make_dataloader(mode=mode, return_ids=True)
    preds = []

    logging.info("Infering...")

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for images in tqdm(test_dataloader):
            images = images.to(device)

            logits = model(images)

            preds.append(torch.argmax(logits, dim=1).cpu())

    logging.info("Saving submission...")

    with open("data/interim/id_to_label.json", "r+") as file:
        id_to_label = json.load(file)

    pd.DataFrame(
        {
            "image_id": image_ids,
            "label": [id_to_label[str(int(id_))] for id_ in torch.cat(preds)],
        }
    ).to_csv("models/submission.csv", index=False)

    logging.info("Submission saved.")
