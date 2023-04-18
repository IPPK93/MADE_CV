import json
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)


def generate() -> None:
    logging.info("Loading labels...")

    df = pd.read_csv("data/raw/train.csv")

    logging.info("Generating id<->label maps...")

    label_to_id = {label: id_ for id_, label in enumerate(set(df["label"]))}
    id_to_label = {id_: label for label, id_ in label_to_id.items()}

    logging.info("Saving maps...")

    with open("data/interim/id_to_label.json", "w+") as file:
        json.dump(id_to_label, file)
    with open("data/interim/label_to_id.json", "w+") as file:
        json.dump(label_to_id, file)

    logging.info("All done!")
