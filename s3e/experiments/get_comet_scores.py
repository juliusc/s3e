"""Get COMET scores for the 'mt' dataset"""
"""Compare uncertainty metrics"""
import argparse
import json
import logging
import math
import sys

from importlib import resources
from pathlib import Path
from tqdm import tqdm

import comet
import h5py
import numpy as np
import torch

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from s3e.lib import utils


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("s3e.get_comet_scores")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = comet.download_model(args.model)
    model = comet.load_from_checkpoint(model_path).eval().to(device)

    for ds_name in ["mt", "qe"]:
        for split in ["validation", "test"]:
            logger.info(f"Computing COMET scores for mt, split '{split}'")

            ds_path = resources.files(f"s3e.data.{ds_name}.{args.language_pair}").joinpath(f"{split}.js")
            dataset = json.load(open(ds_path))

            with (h5py.File(Path(args.work_dir) / f"{ds_name}.{args.language_pair}.{split}.h5", "a")) as h5_file:
                pred_texts = [t.decode() for t in h5_file[utils.PRED_TEXTS_H5DS_NAME]]
                inputs = [{"src": row["src"], "mt": pred}
                          for row, pred in zip(dataset, pred_texts)]

                with torch.no_grad():
                    scores = model.predict(samples=inputs).scores

                pred_sims_bert_h5ds = h5_file.create_dataset(utils.COMET_SCORES_H5DS_NAME, (len(pred_texts),), float)
                pred_sims_bert_h5ds[:] = scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "language_pair",
        help="Language pair. Supported values: 'de-en', 'et-en', 'ne-en'.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "--model", default="Unbabel/wmt22-cometkiwi-da",
        help="COMET model on HuggingFace.")

    args = parser.parse_args()
    main(args)
