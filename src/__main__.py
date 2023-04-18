import argparse

from src.models.train_model import train_model
from src.models.predict_model import predict_model
from src.data.generate_id_label_pairs import generate


def main(args: argparse.Namespace) -> None:
    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        predict_model(args.model_path)
    elif args.mode == "generate":
        generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="EfficientNet V2 M Model")

    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        help='Either "train", "test" or "generate".',
        required=True,
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        help='Path to the trained model. Used only if mode is "test".',
        required=False,
    )

    args = parser.parse_args()

    main(args)
