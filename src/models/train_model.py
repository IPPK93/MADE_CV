import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from tqdm import tqdm

from src.models.create_model import create_model
from src.data.make_dataloader import make_dataloader

import logging
from typing import List

logging.basicConfig(level=logging.INFO)


def _train(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: "torch.optim.lr_scheduler",
    device: torch.device,
    num_epochs: int = 1,
    max_iters: int = 100,
    calc_f1: bool = True,
) -> None:
    writer = SummaryWriter(log_dir="reports/runs", flush_secs=30)

    model.train()
    i = 0
    for _ in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for _, (images, labels) in tqdm(
            zip(range(max_iters), dataloader), total=max_iters
        ):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)
            writer.add_scalar("Loss / Train / Iteration", loss.item(), i)

            if calc_f1:
                preds = torch.argmax(logits, dim=1)
                f_score = f1_score(
                    labels.cpu().data, preds.cpu().data, average="micro"
                )
                writer.add_scalar("F-score / Train / Iteration", f_score, i)

            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()

            # Note that scheduler will be called each iteration
            # instead of each epoch
            scheduler.step()
            i += 1


def _add_classifier(model: nn.Module) -> None:
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features=1280, out_features=320, bias=True),
        nn.SiLU(inplace=True),
        nn.Linear(in_features=320, out_features=30, bias=True),
    )

    nn.init.xavier_normal_(model.classifier[1][0].weight)
    nn.init.xavier_normal_(model.classifier[1][2].weight)


def _freeze_all_layers(model: nn.Module) -> None:
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


def _unfreeze_model_layers(layers: List[nn.Module]) -> None:
    for child in layers:
        for param in child.parameters():
            param.requires_grad = True


def train_model() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading EfficientNet_V2_M model...")

    model = create_model()
    _add_classifier(model)
    model = model.to(device)

    _freeze_all_layers(model)

    logging.info("Loading train dataset...")

    dataloader = make_dataloader(mode="train")

    logging.info("Setting up optimizer and stuff...")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-3, betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5
    )
    criterion = nn.CrossEntropyLoss()

    layers_to_unfreeze = [
        [model.classifier],
        [model.features[8], model.features[7]],
        [model.features[6], model.features[5]],
    ]

    max_iters_list = [250, 300, 250]

    logging.info("Training model...")

    for layers, max_iters in zip(layers_to_unfreeze, max_iters_list):
        _unfreeze_model_layers(layers=layers)
        _train(
            dataloader,
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            max_iters=max_iters,
        )

    logging.info("Model trained. Saving model...")

    torch.save(model, "models/efficientnet_v2_m_clf-87-65-model.pth")

    logging.info("Model saved.")
