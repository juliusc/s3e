import argparse
import json
import logging
import sys

from tqdm import tqdm
from importlib import resources
from pathlib import Path

import h5py
import numpy as np
import torch
import sacrebleu

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from s3e.lib import utils, uncertainty


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("s3e.get_similarities")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for ds_name in ["mt", "qe"]:
        for split in ["validation", "test"]:
            logger.info(f"Computing BERT and chrF++ similarities for dataset '{ds_name}', split '{split}'")

            with (h5py.File(Path(args.work_dir) / f"{ds_name}.{args.language_pair}.{split}.h5", "a")) as h5_file:
                if ds_name == "mt":
                    pred_texts = [t.decode() for t in h5_file[utils.PRED_TEXTS_H5DS_NAME]]
                else:
                    qe_ds = json.load(open(resources.files(f"s3e.data.qe.{args.language_pair}").joinpath(f"{split}.js")))
                    pred_texts = [row["tgt"] for row in qe_ds]

                for h5ds_name in [utils.PRED_SIMS_BERT_H5DS_NAME,
                                  utils.SAMPLE_SIMS_BERT_H5DS_NAME,
                                  utils.PRED_SIMS_CHRF_H5DS_NAME,
                                  utils.SAMPLE_SIMS_CHRF_H5DS_NAME]:
                    if h5ds_name in h5_file:
                        del h5_file[h5ds_name]

                pred_sims_bert_h5ds = h5_file.create_dataset(
                    utils.PRED_SIMS_BERT_H5DS_NAME, (len(pred_texts),), utils.H5_VLEN_FLOAT_DTYPE)
                sample_sims_bert_h5ds = h5_file.create_dataset(
                    utils.SAMPLE_SIMS_BERT_H5DS_NAME, (len(pred_texts),), utils.H5_VLEN_FLOAT_DTYPE)
                pred_sims_chrf_h5ds = h5_file.create_dataset(
                    utils.PRED_SIMS_CHRF_H5DS_NAME, (len(pred_texts),), utils.H5_VLEN_FLOAT_DTYPE)
                sample_sims_chrf_h5ds = h5_file.create_dataset(
                    utils.SAMPLE_SIMS_CHRF_H5DS_NAME, (len(pred_texts),), utils.H5_VLEN_FLOAT_DTYPE)

                for i in tqdm(range(h5_file[utils.SAMPLE_TEXTS_H5DS_NAME].shape[0])):
                    num_unique_samples = (h5_file[utils.SAMPLE_COUNTS_H5_NAME][i] > 0).sum()
                    sample_texts = [t.decode() for _, t in zip(range(num_unique_samples), h5_file[utils.SAMPLE_TEXTS_H5DS_NAME][i, :])]
                    all_texts = list(set([pred_texts[i]] + sample_texts))
                    text_to_idx = dict((text, i) for i, text in enumerate(all_texts))

                    model_inputs = tokenizer(all_texts, padding=True, return_tensors="pt").to(device)
                    embeddings = model(**model_inputs).pooler_output
                    embeddings = F.normalize(embeddings)

                    pred_emb = embeddings[text_to_idx[pred_texts[i]]]
                    sample_idxs = torch.tensor([text_to_idx[text] for text in sample_texts])
                    sample_embs = embeddings[sample_idxs]

                    pred_bert_sims = uncertainty.get_emb_cosine_similarity_matrix(pred_emb.unsqueeze(0), sample_embs).squeeze()
                    pred_sims_bert_h5ds[i] = pred_bert_sims.detach().cpu().tolist()

                    sample_bert_sims = uncertainty.get_emb_cosine_similarity_matrix(sample_embs, sample_embs)
                    sample_sims_bert_h5ds[i] = sample_bert_sims.reshape(-1).detach().cpu().tolist()

                    pred_chrf_sims = uncertainty.get_chrf_similarity_matrix([pred_texts[i]], sample_texts).reshape(-1)
                    pred_sims_chrf_h5ds[i] = pred_chrf_sims

                    sample_chrf_sims = uncertainty.get_chrf_similarity_matrix(sample_texts, sample_texts)
                    sample_sims_chrf_h5ds[i] = sample_chrf_sims.reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "language_pair",
        help="Language pair. Supported values: 'de-en', 'et-en', 'ne-en'.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "--model", default="princeton-nlp/sup-simcse-roberta-large",
        help="HuggingFace NMT model. Only NLLB models supported because other "
             "models have different interfaces for language setting.")

    args = parser.parse_args()
    main(args)
