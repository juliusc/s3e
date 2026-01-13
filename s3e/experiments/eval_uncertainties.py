"""Compare uncertainty metrics"""
import argparse
import itertools
import json
import logging
import math
import sys

from importlib import resources
from pathlib import Path
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import torch

from scipy.stats import pearsonr, kendalltau

from s3e.lib import utils, uncertainty


def get_best_alpha(uncertainties, gold_scores, alphas, corr_func, max_or_min="max"):
    corrs = []
    for i in range(uncertainties.shape[1]):
        corrs.append(-corr_func(uncertainties[:, i], gold_scores).statistic)
    if max_or_min == "max":
        return alphas[np.array(corrs).argmax()]
    else:
        return alphas[np.array(corrs).argmin()]


def get_avg_token_stats(row, counts, avg_or_sum="avg"):
    """Get the average of a token statistic, such as the mean token logprob.
    
    The stat is first aggregated within a sample using avg_or_sum, then averaged across samples.
    """
    stats = np.zeros(row.shape)
    for i in range(counts.shape[0]):
        if counts[i] == 0:
            break
        if avg_or_sum == "sum":
            stats[i] = row[i].sum() * counts[i]
        else:
            stats[i] = row[i].mean() * counts[i]
    return stats.sum() / counts.sum()


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("s3e.eval_uncertainties")

    alphas = [float(s) for s in args.alphas.split(",")]

    printouts = []

    for ds_name in ["mt", "qe"]:
        logger.info(f"Searching optimal alpha for dataset '{ds_name}' on validation set.")

        with (h5py.File(Path(args.work_dir) / f"{ds_name}.{args.language_pair}.validation.h5")) as h5_file:
            num_instances = h5_file[utils.SAMPLE_TEXTS_H5DS_NAME].shape[0]
            sss_chrf = np.zeros((num_instances, len(alphas)))
            sss_bert = np.zeros(sss_chrf.shape)
            s3e_chrf = np.zeros(sss_chrf.shape)
            s3e_bert = np.zeros(sss_chrf.shape)

            for i in tqdm(range(num_instances)):
                num_unique_samples = (h5_file[utils.SAMPLE_COUNTS_H5_NAME][i] > 0).sum()
                pred_seq_logprob = torch.tensor(h5_file[utils.PRED_TOKEN_LOGPROBS_H5DS_NAME][i]).sum()
                sample_seq_logprobs = torch.tensor([x.sum() for x in h5_file[utils.SAMPLE_TOKEN_LOGPROBS_H5DS_NAME][i, :num_unique_samples]])
                counts = torch.tensor(h5_file[utils.SAMPLE_COUNTS_H5_NAME][i], dtype=torch.float)
                
                pred_tokens = tuple(h5_file[utils.PRED_TOKENS_H5DS_NAME][i])
                sample_tokens = [tuple(x) for x in h5_file[utils.SAMPLE_TOKENS_H5DS_NAME][i, :num_unique_samples]]

                pred_sims_chrf = torch.tensor(h5_file[utils.PRED_SIMS_CHRF_H5DS_NAME][i])
                pred_sims_bert = torch.tensor(h5_file[utils.PRED_SIMS_BERT_H5DS_NAME][i])
                sample_sims_chrf = utils.reshape_similarity_matrix(torch.tensor(h5_file[utils.SAMPLE_SIMS_CHRF_H5DS_NAME][i]))
                sample_sims_bert = utils.reshape_similarity_matrix(torch.tensor(h5_file[utils.SAMPLE_SIMS_BERT_H5DS_NAME][i]))
                for j, alpha in enumerate(alphas):
                    sss_chrf[i, j] = uncertainty.compute_sss(pred_tokens, sample_tokens, pred_seq_logprob, pred_sims_chrf, counts, alpha=alpha)
                    sss_bert[i, j] = uncertainty.compute_sss(pred_tokens, sample_tokens, pred_seq_logprob, pred_sims_bert, counts, alpha=alpha)

                    s3e_chrf[i, j] = uncertainty.compute_s3e(sample_seq_logprobs, sample_sims_chrf, counts, alpha=alpha)
                    s3e_bert[i, j] = uncertainty.compute_s3e(sample_seq_logprobs, sample_sims_bert, counts, alpha=alpha)

            if ds_name == "mt":
                gold_scores = h5_file[utils.COMET_SCORES_H5DS_NAME][:]
            else:
                ds_path = resources.files(f"s3e.data.{ds_name}.{args.language_pair}").joinpath(f"{split}.js")
                gold_scores = [row["score"] for row in json.load(open(ds_path))[:num_instances]]

        best_alpha_sss_chrf_r = get_best_alpha(sss_chrf, gold_scores, alphas, pearsonr)
        best_alpha_sss_chrf_tau = get_best_alpha(sss_chrf, gold_scores, alphas, kendalltau)

        best_alpha_sss_bert_r = get_best_alpha(sss_bert, gold_scores, alphas, pearsonr)
        best_alpha_sss_bert_tau = get_best_alpha(sss_bert, gold_scores, alphas, kendalltau)

        if ds_name == "mt":
            best_alpha_s3e_chrf_r = get_best_alpha(s3e_chrf, gold_scores, alphas, pearsonr)
            best_alpha_s3e_chrf_tau = get_best_alpha(s3e_chrf, gold_scores, alphas, kendalltau)

            best_alpha_s3e_bert_r = get_best_alpha(s3e_bert, gold_scores, alphas, pearsonr)
            best_alpha_s3e_bert_tau = get_best_alpha(s3e_bert, gold_scores, alphas, kendalltau)

        split = "test"
        logger.info(f"Computing uncertainty correlations for '{ds_name}', split '{split}'")

        with (h5py.File(Path(args.work_dir) / f"{ds_name}.{args.language_pair}.{split}.h5")) as h5_file:
            # Similarity-sensitive uncertainties
            sss_chrf_best_r = np.zeros(h5_file[utils.SAMPLE_TEXTS_H5DS_NAME].shape[0])
            sss_chrf_best_tau = np.zeros(sss_chrf_best_r.shape)
            sss_bert_best_r = np.zeros(sss_chrf_best_r.shape)
            sss_bert_best_tau = np.zeros(sss_chrf_best_r.shape)
            s3e_chrf_best_r = np.zeros(sss_chrf_best_r.shape)
            s3e_chrf_best_tau = np.zeros(sss_chrf_best_r.shape)
            s3e_bert_best_r = np.zeros(sss_chrf_best_r.shape)
            s3e_bert_best_tau = np.zeros(sss_chrf_best_r.shape)

            # Baseline uncertainties
            pred_avg_token_logprob = np.zeros(sss_chrf_best_r.shape)
            pred_total_token_logprob = np.zeros(sss_chrf_best_r.shape)
            pred_avg_token_entropy = np.zeros(sss_chrf_best_r.shape)
            pred_total_token_entropy = np.zeros(sss_chrf_best_r.shape)            

            sample_avg_token_logprob = np.zeros(sss_chrf_best_r.shape)
            sample_total_token_logprob = np.zeros(sss_chrf_best_r.shape)
            sample_avg_token_entropy = np.zeros(sss_chrf_best_r.shape)
            sample_total_token_entropy = np.zeros(sss_chrf_best_r.shape)            

            for i in tqdm(range(h5_file[utils.SAMPLE_TEXTS_H5DS_NAME].shape[0])):
                num_unique_samples = (h5_file[utils.SAMPLE_COUNTS_H5_NAME][i] > 0).sum()
                pred_seq_logprob = torch.tensor(h5_file[utils.PRED_TOKEN_LOGPROBS_H5DS_NAME][i]).sum()
                sample_seq_logprobs = torch.tensor([x.sum() for x in h5_file[utils.SAMPLE_TOKEN_LOGPROBS_H5DS_NAME][i, :num_unique_samples]])
                counts = torch.tensor(h5_file[utils.SAMPLE_COUNTS_H5_NAME][i], dtype=torch.float)
                
                pred_tokens = tuple(h5_file[utils.PRED_TOKENS_H5DS_NAME][i])
                sample_tokens = [tuple(x) for x in h5_file[utils.SAMPLE_TOKENS_H5DS_NAME][i, :num_unique_samples]]

                pred_sims_chrf = torch.tensor(h5_file[utils.PRED_SIMS_CHRF_H5DS_NAME][i])
                pred_sims_bert = torch.tensor(h5_file[utils.PRED_SIMS_BERT_H5DS_NAME][i])
                sample_sims_chrf = utils.reshape_similarity_matrix(torch.tensor(h5_file[utils.SAMPLE_SIMS_CHRF_H5DS_NAME][i]))
                sample_sims_bert = utils.reshape_similarity_matrix(torch.tensor(h5_file[utils.SAMPLE_SIMS_BERT_H5DS_NAME][i]))

                sss_chrf_best_r[i] = uncertainty.compute_sss(pred_tokens, sample_tokens, pred_seq_logprob, pred_sims_chrf, counts, alpha=best_alpha_sss_chrf_r)
                sss_chrf_best_tau[i] = uncertainty.compute_sss(pred_tokens, sample_tokens, pred_seq_logprob, pred_sims_chrf, counts, alpha=best_alpha_sss_chrf_tau)

                sss_bert_best_r[i] = uncertainty.compute_sss(pred_tokens, sample_tokens, pred_seq_logprob, pred_sims_bert, counts, alpha=best_alpha_sss_bert_r)
                sss_bert_best_tau[i] = uncertainty.compute_sss(pred_tokens, sample_tokens, pred_seq_logprob, pred_sims_bert, counts, alpha=best_alpha_sss_bert_tau)

                if ds_name == "mt":
                    s3e_chrf_best_r[i] = uncertainty.compute_s3e(sample_seq_logprobs, sample_sims_chrf, counts, alpha=best_alpha_s3e_chrf_r)
                    s3e_chrf_best_tau[i] = uncertainty.compute_s3e(sample_seq_logprobs, sample_sims_chrf, counts, alpha=best_alpha_s3e_chrf_tau)

                    s3e_bert_best_r[i] = uncertainty.compute_s3e(sample_seq_logprobs, sample_sims_bert, counts, alpha=best_alpha_s3e_bert_r)
                    s3e_bert_best_tau[i] = uncertainty.compute_s3e(sample_seq_logprobs, sample_sims_bert, counts, alpha=best_alpha_s3e_bert_tau)

                pred_avg_token_logprob[i] = h5_file[utils.PRED_TOKEN_LOGPROBS_H5DS_NAME][i].mean()
                pred_total_token_logprob[i] = h5_file[utils.PRED_TOKEN_LOGPROBS_H5DS_NAME][i].sum()
                pred_avg_token_entropy[i] = h5_file[utils.PRED_TOKEN_ENTROPIES_H5DS_NAME][i].mean()
                pred_total_token_entropy[i] = h5_file[utils.PRED_TOKEN_ENTROPIES_H5DS_NAME][i].sum()

                if ds_name == "mt":
                    sample_avg_token_logprob[i] = get_avg_token_stats(h5_file[utils.SAMPLE_TOKEN_LOGPROBS_H5DS_NAME][i, :num_unique_samples], counts, "avg")
                    sample_total_token_logprob[i] = get_avg_token_stats(h5_file[utils.SAMPLE_TOKEN_LOGPROBS_H5DS_NAME][i, :num_unique_samples], counts, "sum")
                    sample_avg_token_entropy[i] = get_avg_token_stats(h5_file[utils.SAMPLE_TOKEN_ENTROPIES_H5DS_NAME][i, :num_unique_samples], counts, "avg")
                    sample_total_token_entropy[i] = get_avg_token_stats(h5_file[utils.SAMPLE_TOKEN_ENTROPIES_H5DS_NAME][i, :num_unique_samples], counts, "sum")

            comet_scores = h5_file[utils.COMET_SCORES_H5DS_NAME][:]
            if ds_name == "mt":
                gold_scores = comet_scores
            else:
                ds_path = resources.files(f"s3e.data.{ds_name}.{args.language_pair}").joinpath(f"{split}.js")
                gold_scores = [row["score"] for row in json.load(open(ds_path))[:num_instances]]

        df = pd.DataFrame(columns=["Uncertainty", "Correlation Type", "Value"])
        if ds_name == "mt":
            df.loc[len(df)] = ["S3E, chrF++", "r", -pearsonr(s3e_chrf_best_r, gold_scores).statistic]
            df.loc[len(df)] = ["S3E, BERT", "r", -pearsonr(s3e_bert_best_r, gold_scores).statistic]

        df.loc[len(df)] = ["SSS, chrF++", "r", -pearsonr(sss_chrf_best_r, gold_scores).statistic]
        df.loc[len(df)] = ["SSS, BERT", "r", -pearsonr(sss_bert_best_r, gold_scores).statistic]

        df.loc[len(df)] = ["Pred. avg. token logprob", "r", pearsonr(pred_avg_token_logprob, gold_scores).statistic]
        df.loc[len(df)] = ["Pred. seq logprob", "r", pearsonr(pred_total_token_logprob, gold_scores).statistic]
        df.loc[len(df)] = ["Pred. avg. token entropy", "r", -pearsonr(pred_avg_token_entropy, gold_scores).statistic]
        df.loc[len(df)] = ["Pred. total token entropy", "r", -pearsonr(pred_total_token_entropy, gold_scores).statistic]

        if ds_name == "qe":
            df.loc[len(df)] = ["COMET", "r", pearsonr(comet_scores, gold_scores).statistic]

        if ds_name == "mt":
            df.loc[len(df)] = ["S3E, chrF++", "tau", -kendalltau(s3e_chrf_best_tau, gold_scores).statistic]
            df.loc[len(df)] = ["S3E, BERT", "tau", -kendalltau(s3e_bert_best_tau, gold_scores).statistic]

        df.loc[len(df)] = ["SSS, chrF++", "tau", -kendalltau(sss_chrf_best_tau, gold_scores).statistic]
        df.loc[len(df)] = ["SSS, BERT", "tau", -kendalltau(sss_bert_best_tau, gold_scores).statistic]

        df.loc[len(df)] = ["Pred. avg. token logprob", "tau", kendalltau(pred_avg_token_logprob, gold_scores).statistic]
        df.loc[len(df)] = ["Pred. seq logprob", "tau", kendalltau(pred_total_token_logprob, gold_scores).statistic]
        df.loc[len(df)] = ["Pred. avg. token entropy", "tau", -kendalltau(pred_avg_token_entropy, gold_scores).statistic]
        df.loc[len(df)] = ["Pred. total token entropy", "tau", -kendalltau(pred_total_token_entropy, gold_scores).statistic]

        if ds_name == "qe":
            df.loc[len(df)] = ["COMET", "tau", kendalltau(comet_scores, gold_scores).statistic]

        printouts.append(f"*** Correlations on '{ds_name}' dataset ***\n{str(df)}")

    print('\n\n'.join(printouts))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "language_pair",
        help="Language pair. Supported values: 'de-en', 'et-en', 'ne-en'.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "--alphas", default="-1,0,1,2,3,4,5,6,7,8,9,10",
        help="Values of alpha to use in hyperparameter sweep. See paper for definition.") 


    args = parser.parse_args()
    main(args)
