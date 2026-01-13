# TODO: Add logging messages
import argparse
import gc
import itertools
import json
import logging
import math
import sys
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scipy.stats import pearsonr, kendalltau
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from s3e.lib import utils, uncertainty


def batched(iterable: list, n: int):
    iterator = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterator, n))
        if not chunk:
            return
        yield chunk


def generate_predictions(dataset, model, tokenizer, gen_config, h5_file, forced_bos_token_id, batch_size=1):
    text_h5ds = h5_file.create_dataset(utils.PRED_TEXTS_H5DS_NAME, (len(dataset),), utils.H5_STRING_DTYPE)
    tokens_h5ds = h5_file.create_dataset(utils.PRED_TOKENS_H5DS_NAME, (len(dataset),), utils.H5_VLEN_INT_DTYPE)
    token_logprob_h5ds = h5_file.create_dataset(
        utils.PRED_TOKEN_LOGPROBS_H5DS_NAME, (len(dataset),), utils.H5_VLEN_FLOAT_DTYPE)
    token_entropy_h5ds = h5_file.create_dataset(
        utils.PRED_TOKEN_ENTROPIES_H5DS_NAME, (len(dataset),), utils.H5_VLEN_FLOAT_DTYPE)

    row_id = 0
    for batch in tqdm(batched(dataset, batch_size),
                      total=int(math.ceil(len(dataset) / batch_size))):
        src_sents = [row["src"] for row in batch]
        inputs = tokenizer(src_sents, padding=True, return_tensors="pt")
        inputs.to(model.device)
        output = model.generate( **inputs, forced_bos_token_id=forced_bos_token_id, generation_config=gen_config, return_dict_in_generate=True, output_scores=True, renormalize_logits=True)

        texts, all_tokens, all_token_logprobs, all_token_entropies = utils.process_generation_output(output, tokenizer)
        for i in range(len(batch)):
            text_h5ds[row_id+i] = texts[i]
            tokens_h5ds[row_id+i] = all_tokens[i]
            token_logprob_h5ds[row_id+i] = all_token_logprobs[i]
            token_entropy_h5ds[row_id+i] = all_token_entropies[i]
        row_id += len(batch)


def generate_samples(dataset, model, tokenizer, gen_config, h5_file, forced_bos_token_id, num_samples=64, batch_size=64):
    text_h5ds = h5_file.create_dataset(utils.SAMPLE_TEXTS_H5DS_NAME, (len(dataset), num_samples), utils.H5_STRING_DTYPE)
    tokens_h5ds = h5_file.create_dataset(utils.SAMPLE_TOKENS_H5DS_NAME, (len(dataset), num_samples), utils.H5_VLEN_INT_DTYPE)
    logprobs_h5ds = h5_file.create_dataset(
        utils.SAMPLE_TOKEN_LOGPROBS_H5DS_NAME, (len(dataset), num_samples), float)
    count_h5ds = h5_file.create_dataset(utils.SAMPLE_COUNTS_H5_NAME, (len(dataset),), utils.H5_VLEN_INT_DTYPE)

    instances_per_call = max(1, int(batch_size / num_samples))
    calls_per_instance = max(1, int(num_samples / batch_size))

    row_idx = 0
    for batch in tqdm(batched(dataset, instances_per_call),
                      total=int(math.ceil(len(dataset) / instances_per_call))):
        src_sents = [row["src"] for row in batch]
        inputs = tokenizer(src_sents, padding=True, return_tensors="pt")
        inputs.to(model.device)
        
        all_texts = []
        all_tokens = []
        all_token_logprobs = []
        for _ in range(calls_per_instance):
            output = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, generation_config=gen_config,
                return_dict_in_generate=True, output_scores=True, renormalize_logits=True)
            texts, tokens, token_logprobs, _ = utils.process_generation_output(output, tokenizer)
            all_texts.extend(texts)
            all_tokens.extend(tokens)
            all_token_logprobs.extend(token_logprobs)

        for instance_idx in range(instances_per_call):
            start_idx = instance_idx * num_samples
            if start_idx > len(all_texts):
                break
            end_idx = (instance_idx + 1) * num_samples
            texts_counter = Counter(all_texts[start_idx:end_idx])
            texts_to_idx = dict((text, i) for i, text in enumerate(all_texts))
            counts = []
            for sample_idx, (text, count) in enumerate(sorted(texts_counter.items(), key=lambda x: -x[1])):
                text_h5ds[row_idx + instance_idx, sample_idx] = text
                tokens_h5ds[row_idx + instance_idx, sample_idx] = all_tokens[texts_to_idx[text]]
                logprobs_h5ds[row_idx + instance_idx, sample_idx] = sum(all_token_logprobs[texts_to_idx[text]])
                counts.append(count)
            count_h5ds[row_idx + instance_idx] = counts

        row_idx += len(batch)


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("s3e.generate_sequences")

    if (args.num_samples % args.sampling_batch_size and
        args.sampling_batch_size % args.num_samples):
        raise ValueError("--num_samples must be a multiple of or divisible by --sampling_batch_size")

    if args.mode == "validation":
        alphas = [int(x) for x in args.alphas_sweep.split(",")]
        if args.corr_func == "pearsonr":
            corr_func = pearsonr
        elif args.corr_func == "kendalltau":
            corr_func = kendalltau
        else:
            raise ValueError("--corr_func must be 'pearsonr' or 'kendalltau'")
    elif args.mode == "test":
        alphas = [args.alpha]
    else:
        raise ValueError(f"--mode must be 'test' or 'validation'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model)
    gen_model.eval()
    gen_model.to(device)
    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_model, src_lang=args.src_lang_code, tgt_lang=args.tgt_lang_code)
    
    forced_bos_id = gen_tokenizer.convert_tokens_to_ids(args.tgt_lang_code)

    pred_gen_config = GenerationConfig(
        max_length=args.max_generation_length,
        num_beams=args.prediction_beam_size,
        num_return_sequences=1,
        early_stopping=True,
        do_sample=False
    )
    sample_gen_config = GenerationConfig(
        max_length=args.max_generation_length,
        num_beams=1,
        num_return_sequences=min(args.num_samples, args.sampling_batch_size),
        do_sample=True,
        epsilon_cutoff=args.sampling_epsilon
    )

    dataset = json.load(open(args.dataset_path))
    if args.subset:
        dataset = dataset[:args.subset]

    with h5py.File(args.output_file, "w") as h5_file:
        logger.info(f"Generating predictions...")
        generate_predictions(
            dataset, gen_model, gen_tokenizer, pred_gen_config, h5_file, forced_bos_id,
            batch_size=args.prediction_batch_size)
        logger.info(f"Generating samples...")
        generate_samples(
            dataset, gen_model, gen_tokenizer, sample_gen_config, h5_file, forced_bos_id,
            num_samples=args.num_samples, batch_size=args.sampling_batch_size)

        del gen_model, gen_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        similarities_h5ds = h5_file.create_dataset(
            "similarities", (len(dataset),), utils.H5_VLEN_FLOAT_DTYPE)
        uncertainties_h5ds = h5_file.create_dataset(
            "uncertainties", (len(dataset), len(alphas)), float)
        alphas_h5ds = h5_file.create_dataset("alphas", (len(alphas),), float)
        alphas_h5ds[:] = alphas

        logger.info(f"Computing similarities...")
        if args.sim_func == 'BERT':
            bert_model = AutoModel.from_pretrained(args.bert_model)
            bert_model.eval()
            bert_model.to(device)
            bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

        for i in tqdm(range(len(dataset))):
            num_unique_samples = (h5_file[utils.SAMPLE_COUNTS_H5_NAME][i] > 0).sum()
            sample_texts = [t.decode() for _, t in zip(range(num_unique_samples), h5_file[utils.SAMPLE_TEXTS_H5DS_NAME][i, :])]

            if args.sim_func == 'BERT':
                if args.uncertainty == "SSS":
                    pred_text = h5_file[utils.SAMPLE_TEXTS_H5DS_NAME][i]
                    all_texts = list(set([pred_text] + sample_texts))
                else:
                    all_texts = sample_texts
                    
                text_to_idx = dict((text, i) for i, text in enumerate(all_texts))
                model_inputs = bert_tokenizer(all_texts, padding=True, return_tensors="pt").to(device)

                embeddings = bert_model(**model_inputs).pooler_output
                embeddings = F.normalize(embeddings)

                sample_idxs = torch.tensor([text_to_idx[text] for text in sample_texts])
                sample_embs = embeddings[sample_idxs]

                if args.uncertainty == "SSS":
                    pred_emb = embeddings[text_to_idx[pred_text]]
                    sims = uncertainty.get_emb_cosine_similarity_matrix(pred_emb.unsqueeze(0), sample_embs)
                else:
                    sims = uncertainty.get_emb_cosine_similarity_matrix(sample_embs, sample_embs)

            elif args.sim_func == "chrF":
                if args.uncertainty == "SSS":
                    pred_text = h5_file[utils.SAMPLE_TEXTS_H5DS_NAME][i]
                    sims = uncertainty.get_chrf_similarity_matrix(pred_text, sample_texts)
                else:
                    sims = uncertainty.get_chrf_similarity_matrix(sample_texts, sample_texts)
            else:
                raise ValueError(f"Arg 'uncertainty' must be either S3E or SSS.")
            similarities_h5ds[i] = sims.reshape(-1).detach().cpu().numpy()

            counts = torch.tensor(h5_file[utils.SAMPLE_COUNTS_H5_NAME][i], dtype=float)
            sample_tokens = [tuple(x) for x in h5_file[utils.SAMPLE_TOKENS_H5DS_NAME][i, :num_unique_samples]]

            for alpha_idx, alpha in enumerate(alphas):
                if args.uncertainty == "SSS":
                    pred_tokens = tuple(h5_file[utils.PRED_TOKENS_H5DS_NAME][i])
                    pred_logprob = h5_file[utils.PRED_TOKEN_LOGPROBS_H5DS_NAME][i].sum()
                    uncertainties_h5ds[i, alpha_idx] = uncertainty.compute_sss(
                        pred_tokens, sample_tokens, pred_logprob, similarities_h5ds[i], counts, alpha=alpha)
                else:
                    sample_logprobs = torch.tensor([x.sum() for x in h5_file[utils.SAMPLE_TOKEN_LOGPROBS_H5DS_NAME][i, :num_unique_samples]])
                    sample_sims = utils.reshape_similarity_matrix(torch.tensor(similarities_h5ds[i]))
                    uncertainties_h5ds[i, alpha_idx] = uncertainty.compute_s3e(sample_logprobs, sample_sims, counts, alpha=alpha)
            

        if args.mode == "test":
            logger.info(f"Done. Inspect dataset 'uncertainties' in h5py file {args.output_file} to retrieve uncertainties")
        else:
            logging.info("Scoring predictions with COMET...")
            import comet
            comet_model_path = comet.download_model(args.comet_model)
            comet_model = comet.load_from_checkpoint(comet_model_path).eval().to(device)

            pred_texts = [t.decode() for t in h5_file[utils.PRED_TEXTS_H5DS_NAME]]
            inputs = [{"src": row["src"], "mt": pred}
                       for row, pred in zip(dataset, pred_texts)]

            with torch.no_grad():
                comet_scores = comet_model.predict(samples=inputs).scores

            df = pd.DataFrame(columns=["Alpha", "Correlation with COMET"])
            corrs = []
            for alpha_idx, alpha in enumerate(alphas):
                corr = corr_func(-uncertainties_h5ds[:, alpha_idx], comet_scores).statistic
                corrs.append(corr)
                df.loc[len(df)] = [alpha, corr]

            print(df.to_string(index=False))
            print(f"Best alpha: {alphas[np.array(corrs).argmax()]}")

            breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_path",
        help="Path to dataset, a JSON file containing of list of dicts, where each dict has key 'src'.")

    parser.add_argument(
        "src_lang_code",
        help="FLORES-101 language code for the source language (https://github.com/facebookresearch/flores/blob/main/flores200/README.md)")

    parser.add_argument(
        "tgt_lang_code",
        help="FLORES-101 language code for the target language")

    parser.add_argument(
        "output_file",
        help="Path to file where results as well as intermediate work are saved.")

    parser.add_argument(
        "--mode", default="test",
        help="'validation' to search for the optimal alpha, or 'test' to compute uncertainties. "
             "If --mode=validation, the dataset file should contain a 'score' key in each row representing a quality rating.")

    parser.add_argument(
        "--alphas_sweep", default="0,1,2,3,4,5,6,7,8,9,10",
        help="Values of alpha to search over if --mode=validation. A comma separated list of numbers.")

    parser.add_argument(
        "--alpha", type=int, default=0,
        help="Value of alpha to use if --mode=test.")

    parser.add_argument(
        "--corr_func", default="kendalltau",
        help="Type of correlation used for optimizing alpha when --mode=validation. Must be either 'kendalltau' or 'pearsonr'."
    )

    parser.add_argument(
        "--gen_model", default="facebook/nllb-200-distilled-1.3B",
        help="HuggingFace NMT model. Only NLLB models supported because other "
             "models have different interfaces for language setting.")

    parser.add_argument(
        "--uncertainty", default="S3E",
        help="Type of similarity-sensitive uncertainty, either 'S3E' or 'SSS'")

    parser.add_argument(
        "--sim_func", default="BERT",
        help="Similarity function, either 'BERT' or 'chrF'")

    parser.add_argument(
        "--bert_model", default="princeton-nlp/sup-simcse-roberta-large",
        help="BERT embedding model to be used if sim_func='BERT'")

    parser.add_argument(
        "--comet_model", default="Unbabel/wmt22-cometkiwi-da",
        help="COMET model used to produce a quality score used in validation. Only used if --mode=validation")

    parser.add_argument(
        "--subset", type=int, help="Only process the first n items of the dataset for debugging")

    # LM generation args for prediction
    parser.add_argument(
        "--prediction_batch_size", type=int, default=16, help="Prediction batch size")

    parser.add_argument(
        "--prediction_beam_size", type=int, default=1, help="Prediction beam size")

    # LM generation args for sampling
    parser.add_argument(
        "--num_samples", type=int, default=64, help="Number of samples")

    parser.add_argument(
        "--sampling_batch_size", type=int, default=64,
        help="How many samples to compute at once. Must be a multiple of or divisible by --num_samples.")

    parser.add_argument(
        "--sampling_epsilon", type=float, default=0.02, help="Epsilon for epsilon-sampling")

    # LM generation args for both sampling and generation
    parser.add_argument(
        "--max_generation_length", type=int, default=1024, help="Max generation length")

    args = parser.parse_args()
    main(args)
