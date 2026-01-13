# TODO: Add logging messages
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from transformers import GenerationConfig

from s3e.lib import generation, datasets

PREDICTION_BATCH_SIZE = 1
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
# NUM_SAMPLES = 8
EPSILON_CUTOFF = 0.02
SAMPLING_BATCH_SIZE = 1
SAMPLING_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_beams=1,
    num_return_sequences=NUM_SAMPLES,
    do_sample=True,
    epsilon_cutoff=EPSILON_CUTOFF
)

H5_STRING_DTYPE = h5py.special_dtype(vlen=str)
H5_VLEN_FLOAT_DTYPE = h5py.vlen_dtype(np.dtype('float32'))


def group_batch(batch):
    return {k: [v] for k, v in batch.items()}


def generate_predictions(dataset, model, tokenizer, output_path):
    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}

    with h5py.File(output_path, "w") as h5_file:
        text_h5ds = h5_file.create_dataset("text", (len(dataset),), H5_STRING_DTYPE)
        token_logprob_h5ds = h5_file.create_dataset("token_logprob", (len(dataset),), H5_VLEN_FLOAT_DTYPE)
        token_entropy_h5ds = h5_file.create_dataset("token_entropy", (len(dataset),), H5_VLEN_FLOAT_DTYPE)

        row_id = 0
        for batch in dataset.map(group_batch, batched=True, batch_size=PREDICTION_BATCH_SIZE):
            input = tokenizer(
                batch["src"], padding=True, return_tensors="pt").to(model.device)
            result = model.generate(
                input["input_ids"],
                generation_config=PREDICTION_GENERATION_CONFIG,
                return_dict_in_generate=True,
                output_scores=True,
                renormalize_logits=True)
            texts = tokenizer.batch_decode(result.sequences, skip_special_tokens=True)

            if model.forced_bos_token_id:
                result.sequences = result.sequences[:, 2:]
                result.beam_indices = result.beam_indices[:, 1:]
                result.scores = result.scores[1:]
            else:
                result.sequences = result.sequences[:, 1:]

            for i in range(result.sequences.shape[0]):
                token_logprobs = []
                token_entropies = []
                for t in range(result.sequences.shape[1]):
                    pred_token_id = result.sequences[i, t]
                    if result.sequences[i, t] == tokenizer.pad_token_id:
                        break
                    else:
                        beam_index = result.beam_indices[i, t]
                        token_log_prob = result.scores[t][beam_index, pred_token_id].item()
                        token_logprobs.append(token_log_prob)

                        log_probs = result.scores[t][beam_index, :]
                        token_entropy = -(log_probs.exp() * log_probs).nan_to_num().sum().item()
                        token_entropies.append(token_entropy)

                text_h5ds[row_id] = texts[i]
                token_logprob_h5ds[row_id] = np.array(token_logprobs)
                token_entropy_h5ds[row_id] = np.array(token_entropies)
                row_id += 1


def generate_samples(dataset, model, tokenizer, output_path):
    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}

    with h5py.File(output_path, "w") as h5_file:
        text_h5ds = h5_file.create_dataset("text", (len(dataset), NUM_SAMPLES), H5_STRING_DTYPE)
        token_logprob_h5ds = h5_file.create_dataset("token_logprob", (len(dataset), NUM_SAMPLES), H5_VLEN_FLOAT_DTYPE)
        token_entropy_h5ds = h5_file.create_dataset("token_entropy", (len(dataset), NUM_SAMPLES), H5_VLEN_FLOAT_DTYPE)
        count_h5ds = h5_file.create_dataset("count", (len(dataset), NUM_SAMPLES), int)

        row_id = 0
        for batch in dataset.map(group_batch, batched=True, batch_size=SAMPLING_BATCH_SIZE):
            input = tokenizer(
                batch["src"], padding=True, return_tensors="pt").to(model.device)
            result = model.generate(
                input["input_ids"],
                generation_config=SAMPLING_GENERATION_CONFIG,
                return_dict_in_generate=True,
                output_scores=True,
                renormalize_logits=True)
            texts = tokenizer.batch_decode(result.sequences, skip_special_tokens=True)

            if model.forced_bos_token_id:
                result.sequences = result.sequences[:, 2:]
                result.scores = result.scores[1:]
            else:
                result.sequences = result.sequences[:, 1:]

            seen_texts_to_columns = {}

            for batch_row_id in range(len(batch["src"])):
                for sample_id in range(NUM_SAMPLES):
                    i = batch_row_id * SAMPLING_BATCH_SIZE + sample_id
                    text = texts[i]
                    if text in seen_texts_to_columns:
                        count_h5ds[row_id, seen_texts_to_columns[text]] += 1
                    else:
                        token_logprobs = []
                        token_entropies = []
                        for t in range(result.sequences.shape[1]):
                            pred_token_id = result.sequences[i, t]
                            if result.sequences[i, t] == tokenizer.pad_token_id:
                                break
                            else:
                                token_log_prob = result.scores[t][i, pred_token_id].item()
                                token_logprobs.append(token_log_prob)

                                log_probs = result.scores[t][i, :]
                                token_entropy = -(log_probs.exp() * log_probs).nan_to_num().sum().item()
                                token_entropies.append(token_entropy)

                        num_seen_texts = len(seen_texts_to_columns)
                        text_h5ds[row_id, num_seen_texts] = texts[i]
                        token_logprob_h5ds[row_id, num_seen_texts] = np.array(token_logprobs)
                        token_entropy_h5ds[row_id, num_seen_texts] = np.array(token_entropies)
                        count_h5ds[row_id, num_seen_texts] = 1
                        seen_texts_to_columns[text] = len(seen_texts_to_columns)

                row_id += 1
                import pdb; pdb.set_trace()


def generate_predictions(dataset, model, tokenizer, h5_file):
    text_h5ds = h5_file.create_dataset("text", (len(dataset),), H5_STRING_DTYPE)
    token_logprob_h5ds = h5_file.create_dataset("token_logprob", (len(dataset),), H5_VLEN_FLOAT_DTYPE)
    token_entropy_h5ds = h5_file.create_dataset("token_entropy", (len(dataset),), H5_VLEN_FLOAT_DTYPE)

    row_id = 0
    for batch in dataset.map(group_batch, batched=True, batch_size=PREDICTION_BATCH_SIZE):
        inputs = tokenizer(
            batch["src"], padding=True, return_tensors="pt").to(model.device)
        result = model.generate(
            inputs["input_ids"],
            generation_config=PREDICTION_GENERATION_CONFIG,
            # return_dict_in_generate=True,
            return_dict_in_generate=False,
            output_scores=True,
            renormalize_logits=True)

        texts = tokenizer.batch_decode(result.sequences, skip_special_tokens=True)

        if model.forced_bos_token_id:
            result.sequences = result.sequences[:, 2:]
            result.beam_indices = result.beam_indices[:, 1:]
            result.scores = result.scores[1:]
        else:
            result.sequences = result.sequences[:, 1:]

        for i in range(result.sequences.shape[0]):
            token_logprobs = []
            token_entropies = []
            for t in range(result.sequences.shape[1]):
                pred_token_id = result.sequences[i, t]
                if result.sequences[i, t] == tokenizer.pad_token_id:
                    break
                else:
                    beam_index = result.beam_indices[i, t]
                    token_log_prob = result.scores[t][beam_index, pred_token_id].item()
                    token_logprobs.append(token_log_prob)

                    log_probs = result.scores[t][beam_index, :]
                    token_entropy = -(log_probs.exp() * log_probs).nan_to_num().sum().item()
                    token_entropies.append(token_entropy)

            text_h5ds[row_id] = texts[i]
            token_logprob_h5ds[row_id] = np.array(token_logprobs)
            token_entropy_h5ds[row_id] = np.array(token_entropies)
            row_id += 1


def main(args):
    work_dir = Path(args.work_dir)
    work_dir.mkdir(exist_ok=True)

    src_lang = args.language_pair[:2]
    tgt_lang = args.language_pair[2:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = generation.load_model_and_tokenizer(src_lang, tgt_lang)
    model.eval()
    model.to(device)

    for split in ["dev", "test"]:
        dataset = datasets.load_dataset("mt", args.language_pair, split, subset=args.subset)

        with h5py.File(work_dir / f"predictions.{split}.h5", "w") as h5_file:
            generate_predictions(dataset, model, tokenizer, h5_file)

        import pdb; pdb.set_trace()



    for split in ["validation", "test"]:
        dataset = datasets.load_dataset(
            args.language_pair, split).map(dataset_map_fn)
        if args.subset:
            dataset = dataset.select(range(args.subset))

        pred_output_path = work_dir / f"predictions.{split}.h5"
        pred_donefile_path = work_dir / f"predictions.{split}.DONE"
        if not pred_donefile_path.exists():
            generate_predictions(dataset, model, tokenizer, pred_output_path)
            pred_donefile_path.touch()

        sample_output_path = work_dir / f"samples.{split}.h5"
        sample_donefile_path = work_dir / f"samples.{split}.DONE"
        if not sample_donefile_path.exists():
            generate_samples(dataset, model, tokenizer, sample_output_path)
            # sample_donefile_path.touch()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "language_pair",
        help="Language pair. Supported values: 'de-en', 'et-en', 'ne-en'.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "--subset", type=int, help="Only process the first n items.")

    args = parser.parse_args()
    main(args)
