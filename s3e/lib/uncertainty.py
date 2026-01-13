import math
import numpy as np
import sacrebleu
import torch

import torch.nn.functional as F


def compute_embeddings(model, tokenizer, sample_texts, pred_text=None):
    all_texts = list(set([pred_text] + sample_texts))
    text_to_idx = dict((text, i) for i, text in enumerate(all_texts))

    model_inputs = tokenizer.batch_encode_plus(all_texts, padding=True, return_tensors="pt").to(model.device)
    embeddings = model(**model_inputs).pooler_output
    embeddings = F.normalize(embeddings)
    return embeddings, text_to_idx


def get_emb_cosine_similarity_matrix(hyp_embs, ref_embs):
    """Compute pairwise cosine similarities between hypothesis embeddings and reference embeddings.

    Args:
        hyp_embs (torch.Tensor): Embeddings of shape (N, embed_dim). Must be L2-normalized.
        ref_embs (torch.Tensor): Embeddings of shape (M, embed_dim). Must be L2-normalized.

    Returns:
        (torch.Tensor): Cosine similarity matrix of shape (N, M), where cell (n, m) is the cosine
        similarity between hyp_embs[n] and ref_embs[m].    
    """
    return torch.clamp(torch.matmul(hyp_embs, ref_embs.T), max=1, min=0)


def get_chrf_similarity_matrix(hyp_texts, ref_texts):
    chrf = sacrebleu.CHRF()
    mat = torch.zeros((len(hyp_texts), len(ref_texts)))
    for i, hyp_text in enumerate(hyp_texts):
        for j, ref_text in enumerate(ref_texts):
            mat[i, j] = chrf.sentence_score(hyp_text, [ref_text]).score / 100
    return mat


def compute_sss(pred_tokens, sample_tokens, pred_logprob, similarities, counts, alpha=0):
    """Compute the similarity-sensitive surprisal (SSS) of a prediction in the efficient method described in the paper.
    
    Args:
        pred_text (tuple of int): Token sequence of the prediction whose SSS is considered.
        sample_tokens (str): Token sequences of random samples drawn from the same distribution
            as pred_text. For better performance, these should be deduped, and their counts passed
            in by the counts arg.
        pred_logprob (float or torch.tensor with 0 dimenions): Sequence log probability
            of pred_text.
        similarities (torch.tensor): 1D vector of similarities of each sample to the prediction.
        counts (torch.tensor): 1D vector of counts of each sample.
        alpha (float): The similarity scaling factor described in the paper.

    Returns:
        (torch.tensor) 0D tensor of the prediction SSS.
    """
    counts = counts.clone().to(similarities.device)
    # Zero out counts of any sample that equal the prediction
    # NOTE: The paper says that you have to remove samples which equal the prediction.
    # However, they should be equal at the token-level and not text level, so technically
    # this results in incorrect edge case. I should fix this.
    for i, ref_tokens in enumerate(sample_tokens):
        if pred_tokens == ref_tokens:
            counts[i] = 0
    # Compute log of average similarities over samples using LogSumExp trick
    log_similarities = (similarities.log() * math.exp(alpha) + counts.log()).logsumexp(0) - counts.sum().log()
    # Combine log similarities with the log probability of the prediction
    sss = -torch.logaddexp(pred_logprob, (-pred_logprob.expm1()).log() + log_similarities)
    return sss
    

# TODO: maybe a batched version of this?
def compute_s3e(logprobs, similarity_matrix, counts, alpha=0):
    """Compute the similarity-sensitive Shannon entropy (S3E) of a distribution in the efficient method described in the paper.

    The shapes of logprobs, similarity_matrix, counts correspond to a list of samples
    which must be deduped, and their respective counts stored in the arg counts.

    Args:
        logprobs (torch.tensor): 1D vector of log probabilities of each sample.
        similarities (torch.tensor): 2D matrix of similarities over the samples. similarities[i, j] contains
            the similarity of the ith sample to the jth sample. Note that similarities do not need to be symmetric,
            so this matrix is not necessarily symmetric.
        counts (torch.tensor): 1D vector of counts of each sample.
        alpha (float): The similarity scaling factor described in the paper.

    Returns:
        (torch.tensor) 0D tensor of the prediction S3E.
    """
    counts_matrix = counts.repeat(counts.shape[0], 1).to(similarity_matrix.device)
    # Repeat the counts vector to be a matrix and zero out the diagonal. This prevents the similarity between
    # a sample and itself from being used in the computation, as per the paper.
    counts_matrix -= torch.eye(counts.shape[0]).to(counts_matrix.device) * counts_matrix
    # Compute log of average similarities over samples using LogSumExp trick
    log_similarities = (similarity_matrix.log() * math.exp(alpha) + counts_matrix.log()).logsumexp(1) - counts_matrix.sum(1).log()
    # Combine log similarities with the log probability of the prediction
    sss = -torch.logaddexp(logprobs, (-logprobs.expm1()).log() + log_similarities)
    # S3E is the weighted average of SSS
    s3e = (counts * sss).sum() / counts.sum()
    return s3e