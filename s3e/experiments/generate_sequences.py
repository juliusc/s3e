# TODO: Add logging messages
import argparse
import itertools
import json
import logging
import math
import sys
from collections import Counter
from importlib import resources
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from s3e.lib import utils

PREDICTION_BATCH_SIZE = 2
# Fixed hyperparameters from paper
MAX_GENERATION_LENGTH = 1024
PREDICTION_BEAM_SIZE = 5
PREDICTION_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_beams=PREDICTION_BEAM_SIZE,
    num_return_sequences=1,
    early_stopping=True
)
NUM_SAMPLES = 128
EPSILON_CUTOFF = 0.02
SAMPLING_BATCH_SIZE = 1
SAMPLING_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_beams=1,
    num_return_sequences=NUM_SAMPLES,
    do_sample=True,
    epsilon_cutoff=EPSILON_CUTOFF
)


def batched(iterable: list, n: int):
    iterator = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterator, n))
        if not chunk:
            return
        yield chunk


def generate_predictions(dataset, model, tokenizer, h5_file, forced_bos_token_id):
    text_h5ds = h5_file.create_dataset(utils.PRED_TEXTS_H5DS_NAME, (len(dataset),), utils.H5_STRING_DTYPE)
    tokens_h5ds = h5_file.create_dataset(utils.PRED_TOKENS_H5DS_NAME, (len(dataset),), utils.H5_VLEN_INT_DTYPE)
    token_logprob_h5ds = h5_file.create_dataset(
        utils.PRED_TOKEN_LOGPROBS_H5DS_NAME, (len(dataset),), utils.H5_VLEN_FLOAT_DTYPE)
    token_entropy_h5ds = h5_file.create_dataset(
        utils.PRED_TOKEN_ENTROPIES_H5DS_NAME, (len(dataset),), utils.H5_VLEN_FLOAT_DTYPE)

    row_id = 0
    for batch in tqdm(batched(dataset, PREDICTION_BATCH_SIZE),
                      total=int(math.ceil(len(dataset) / PREDICTION_BATCH_SIZE))):
        src_sents = [row["src"] for row in batch]
        inputs = tokenizer(src_sents, padding=True, return_tensors="pt")
        inputs.to(model.device)
        output = model.generate( **inputs, forced_bos_token_id=forced_bos_token_id, generation_config=PREDICTION_GENERATION_CONFIG, return_dict_in_generate=True, output_scores=True, renormalize_logits=True)

        texts, all_tokens, all_token_logprobs, all_token_entropies = utils.process_generation_output(output, tokenizer)
        for i in range(len(batch)):
            text_h5ds[row_id+i] = texts[i]
            tokens_h5ds[row_id+i] = all_tokens[i]
            token_logprob_h5ds[row_id+i] = all_token_logprobs[i]
            token_entropy_h5ds[row_id+i] = all_token_entropies[i]
        row_id += len(batch)


def score_predictions(dataset, model, tokenizer, h5_file):
    text_h5ds = h5_file.create_dataset(utils.PRED_TEXTS_H5DS_NAME, (len(dataset),), utils.H5_STRING_DTYPE)
    tokens_h5ds = h5_file.create_dataset(utils.PRED_TOKENS_H5DS_NAME, (len(dataset),), utils.H5_VLEN_INT_DTYPE)
    token_logprob_h5ds = h5_file.create_dataset(
        utils.PRED_TOKEN_LOGPROBS_H5DS_NAME, (len(dataset),), utils.H5_VLEN_FLOAT_DTYPE)
    token_entropy_h5ds = h5_file.create_dataset(
        utils.PRED_TOKEN_ENTROPIES_H5DS_NAME, (len(dataset),), utils.H5_VLEN_FLOAT_DTYPE)

    row_id = 0
    for batch in tqdm(batched(dataset, PREDICTION_BATCH_SIZE),
                      total=int(math.ceil(len(dataset) / PREDICTION_BATCH_SIZE))):
        src_sents = [row["src"] for row in batch]
        tgt_sents = [row["tgt"] for row in batch]
        inputs = tokenizer(src_sents, text_target=tgt_sents, padding=True, return_tensors="pt").to(model.device)
        
        output = model(input_ids=inputs.input_ids, labels=inputs.labels)
        texts, all_tokens, all_token_logprobs, all_token_entropies = utils.process_model_scoring_output(output, inputs.labels, tokenizer)
        
        for i in range(len(batch)):
            text_h5ds[row_id+i] = texts[i]
            tokens_h5ds[row_id+i] = all_tokens[i]
            token_logprob_h5ds[row_id+i] = all_token_logprobs[i]
            token_entropy_h5ds[row_id+i] = all_token_entropies[i]
        
        row_id += len(batch)


def generate_samples(dataset, model, tokenizer, h5_file, forced_bos_token_id):
    text_h5ds = h5_file.create_dataset(utils.SAMPLE_TEXTS_H5DS_NAME, (len(dataset), NUM_SAMPLES), utils.H5_STRING_DTYPE)
    tokens_h5ds = h5_file.create_dataset(utils.SAMPLE_TOKENS_H5DS_NAME, (len(dataset), NUM_SAMPLES), utils.H5_VLEN_INT_DTYPE)
    token_logprob_h5ds = h5_file.create_dataset(
        utils.SAMPLE_TOKEN_LOGPROBS_H5DS_NAME, (len(dataset), NUM_SAMPLES), utils.H5_VLEN_FLOAT_DTYPE)
    token_entropy_h5ds = h5_file.create_dataset(
        utils.SAMPLE_TOKEN_ENTROPIES_H5DS_NAME, (len(dataset), NUM_SAMPLES), utils.H5_VLEN_FLOAT_DTYPE)
    count_h5ds = h5_file.create_dataset(utils.SAMPLE_COUNTS_H5_NAME, (len(dataset),), utils.H5_VLEN_INT_DTYPE)

    row_id = 0
    for batch in tqdm(batched(dataset, SAMPLING_BATCH_SIZE),
                      total=int(math.ceil(len(dataset) / SAMPLING_BATCH_SIZE))):
        src_sents = [row["src"] for row in batch]
        inputs = tokenizer(src_sents, padding=True, return_tensors="pt")
        inputs.to(model.device)
        output = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, generation_config=SAMPLING_GENERATION_CONFIG, return_dict_in_generate=True, output_scores=True, renormalize_logits=True)
        texts, all_tokens, all_token_logprobs, all_token_entropies = utils.process_generation_output(output, tokenizer)
        for i, (sample_texts, sample_tokens, sample_token_logprobs, sample_token_entropies) in enumerate(zip(
            batched(texts, NUM_SAMPLES),
            batched(all_tokens, NUM_SAMPLES),
            batched(all_token_logprobs, NUM_SAMPLES),
            batched(all_token_entropies, NUM_SAMPLES))):
            texts_counter = Counter(sample_texts)
            texts_to_idx = dict((text, i) for i, text in enumerate(sample_texts))
            counts = []
            for j, (text, count) in enumerate(sorted(texts_counter.items(), key=lambda x: -x[1])):
                text_h5ds[row_id+i, j] = text
                tokens_h5ds[row_id+i, j] = sample_tokens[texts_to_idx[text]]
                token_logprob_h5ds[row_id+i, j] = sample_token_logprobs[texts_to_idx[text]]
                token_entropy_h5ds[row_id+i, j] = sample_token_entropies[texts_to_idx[text]]
                counts.append(count)
            count_h5ds[row_id+i] = counts
        row_id += len(batch)


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("s3e.generate_sequences")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)

    src_lang = args.language_pair[:2]
    tgt_lang = args.language_pair[3:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()
    model.to(device)
    src_lang_code = utils.LANG_TO_FLORES_LANG[src_lang]
    tgt_lang_code = utils.LANG_TO_FLORES_LANG[tgt_lang]
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
    
    forced_bos_id = tokenizer.convert_tokens_to_ids(utils.LANG_TO_FLORES_LANG[tgt_lang])

    for ds_name in ["mt", "qe"]:
        for split in ["validation", "test"]:
            ds_path = resources.files(f"s3e.data.{ds_name}.{args.language_pair}").joinpath(f"{split}.js")
            dataset = json.load(open(ds_path))
            if args.subset:
                dataset = dataset[:args.subset]

            with h5py.File(work_dir / f"{ds_name}.{args.language_pair}.{split}.h5", "w") as h5_file:
                if ds_name == "mt":
                    logger.info(f"Generating predictions for dataset {ds_name}, split '{split}'")
                    generate_predictions(dataset, model, tokenizer, h5_file, forced_bos_id)
                else:
                    logger.info(f"Scoring predictions for dataset {ds_name}, split '{split}'")
                    score_predictions(dataset, model, tokenizer, h5_file)

                logger.info(f"Generating samples for dataset {ds_name}, split '{split}'")
                generate_samples(dataset, model, tokenizer, h5_file, forced_bos_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "language_pair",
        help="Language pair. Supported values: 'de-en', 'et-en', 'ne-en'.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "--model", default="facebook/nllb-200-distilled-1.3B",
        help="HuggingFace NMT model. Only NLLB models supported because other "
             "models have different interfaces for language setting.")

    parser.add_argument(
        "--subset", type=int, help="Only process the first n items.")

    args = parser.parse_args()
    main(args)
