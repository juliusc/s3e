Contains scripts that replicate experiments from the paper. Settings are the same as in the paper, except that the default model used for all language pairs is [NLLB 1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B). The relative quality of the model changes resulting correlation values compared to the paper (especially for low-resource languages), but the general trend remains that the similarity-sensitive uncertainties proposed outperform the baselines.

Example run on a single language pair:
```
export WORK_DIR=work
export LANG_PAIR=ne-en # Paper uses de-en, et-en, and ne-en
python s3e/experiments/generate_sequences.py $LANG_PAIR $WORK_DIR # --subset=100
python s3e/experiments/get_comet_scores.py $LANG_PAIR $WORK_DIR
python s3e/experiments/get_similarities.py $LANG_PAIR $WORK_DIR
# Prints out data for Table 1 
python s3e/experiments/eval_uncertainties.py $LANG_PAIR $WORK_DIR
```