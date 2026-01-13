import h5py
import numpy as np

# FLORES language code mapping
LANG_TO_FLORES_LANG = {
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "est_Latn",
    "ne": "npi_Deva"
}

# h5py filenames and dataset names
PRED_TEXTS_H5DS_NAME = "pred_texts"
PRED_TOKENS_H5DS_NAME = "pred_tokens"

SAMPLE_TEXTS_H5DS_NAME = "sample_texts"
SAMPLE_TOKENS_H5DS_NAME = "sample_tokens"
SAMPLE_COUNTS_H5_NAME = "sample_counts"

PRED_TOKEN_LOGPROBS_H5DS_NAME = "pred_token_logprobs"
PRED_TOKEN_ENTROPIES_H5DS_NAME = "pred_token_entropies"
SAMPLE_TOKEN_LOGPROBS_H5DS_NAME = "sample_token_logprobs"
SAMPLE_TOKEN_ENTROPIES_H5DS_NAME = "sample_token_entropies"

PRED_SIMS_BERT_H5DS_NAME = "pred_sims_bert"
PRED_SIMS_CHRF_H5DS_NAME = "pred_sims_chrf"
SAMPLE_SIMS_BERT_H5DS_NAME = "sample_sims_bert"
SAMPLE_SIMS_CHRF_H5DS_NAME = "sample_sims_chrf"

COMET_SCORES_H5DS_NAME = "comet_scores"

# h5py datatypes
H5_STRING_DTYPE = h5py.special_dtype(vlen=str)
H5_VLEN_FLOAT_DTYPE = h5py.vlen_dtype(np.dtype('float32'))
H5_VLEN_INT_DTYPE = h5py.vlen_dtype(np.dtype('int'))


def process_generation_output(generation_output, tokenizer):
    texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    # Ignore start token and forced language ID token if it exists
    num_ignored_tokens = 1
    if generation_output.sequences[0, 1].item() in tokenizer.added_tokens_decoder:
        num_ignored_tokens += 1
    generation_output.sequences = generation_output.sequences[:, num_ignored_tokens:]
    if "beam_indices" in generation_output:
        generation_output.beam_indices = generation_output.beam_indices[:, num_ignored_tokens-1:]
    generation_output.scores = generation_output.scores[num_ignored_tokens-1:]

    all_tokens = []
    all_token_logprobs = []
    all_token_entropies = []
    for i in range(generation_output.sequences.shape[0]):
        all_tokens.append(generation_output.sequences[i, :(generation_output.sequences[i] != 1).sum()].tolist())

        token_logprobs = []
        token_entropies = []
        for t in range(generation_output.sequences.shape[1]):
            pred_token_id = generation_output.sequences[i, t]
            if generation_output.sequences[i, t] == tokenizer.pad_token_id:
                break
            if "beam_indices" in generation_output:
                beam_index = generation_output.beam_indices[i, t]
                token_distribution = generation_output.scores[t][beam_index]
            else:
                token_distribution = generation_output.scores[t][i]
            token_log_prob = token_distribution[pred_token_id].item()
            token_logprobs.append(token_log_prob)

            token_entropy = -(token_distribution.exp() * token_distribution).nan_to_num().sum().item()
            token_entropies.append(token_entropy)

        all_token_logprobs.append(token_logprobs)
        all_token_entropies.append(token_entropies)

    return texts, all_tokens, all_token_logprobs, all_token_entropies


def process_model_scoring_output(model_output, labels, tokenizer):
    texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Ignore start token and forced language ID token if it exists
    num_ignored_tokens = 0
    if labels[0, 0].item() in tokenizer.added_tokens_decoder:
        num_ignored_tokens = 1
    labels = labels[:, num_ignored_tokens:]
    model_output.logits = model_output.logits[:, num_ignored_tokens:]

    all_tokens = []
    all_token_logprobs = []
    all_token_entropies = []
    for i in range(labels.shape[0]):
        all_tokens.append(labels[i, :(labels[i] != 1).sum()].tolist())
        token_logprobs = []
        token_entropies = []
        for t in range(labels.shape[1]):
            pred_token_id = labels[i, t]
            if labels[i, t] == tokenizer.pad_token_id:
                break
            token_distribution = model_output.logits[i][t].log_softmax(0)
            token_log_prob = token_distribution[pred_token_id].item()
            token_logprobs.append(token_log_prob)

            token_entropy = -(token_distribution.exp() * token_distribution).nan_to_num().sum().item()
            token_entropies.append(token_entropy)

        all_token_logprobs.append(token_logprobs)
        all_token_entropies.append(token_entropies)

    return texts, all_tokens, all_token_logprobs, all_token_entropies


def reshape_similarity_matrix(arr):
    dim = int(arr.shape[0] ** 0.5)
    return arr.reshape((dim, dim))
