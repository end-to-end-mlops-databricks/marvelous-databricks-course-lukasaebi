import argparse
import logging

import yaml

from reservations.config import DataConfig
from reservations.data import DataLoader
from reservations.evaluate import Accuracy
from reservations.model import RandomForestModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config-path", dest="data_config_path", type=str, help="Path to the config yaml file.")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, help="Path to the dataset.")
    return parser.parse_args()


def main(args):
    with open(args.data_config_path, "r") as f:
        config = yaml.safe_load(f)

    data_config = DataConfig(**config)
    data_loader = DataLoader(args.dataset_path, data_config)
    X_train, X_test, y_train, y_test = data_loader.preprocess_data()

    model = RandomForestModel()
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    acc = Accuracy.calculate(y_test, y_preds)

    log_msg = f"""Evaluation results:
    Dataset {args.dataset_path}
    Model: {model.__class__.__name__}
    Accuracy: {acc}
    """
    logger.info(log_msg)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
