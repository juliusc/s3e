This repo contains an implementation of similarity-sensitive entropy, proposed by [Measuring Uncertainty in Neural Machine Translation with Similarity-Sensitive Entropy](https://aclanthology.org/2024.eacl-long.129/), by Julius Cheng and Andreas Vlachos, presented at EACL 2024.

Similarity-sensitive entropy extends the classic Shannon entropies to account for similarities between elements. This is helpful for natural language generation tasks such as machine translation, where output distribution probability mass might be distributed across highly similar sequences. Similarity-sensitive entropy, unlike standard entropy, captures whether probability mass is spread across similar or dissimilar sequences, and this can be problematic when using entropy as a measure of uncertainty, spread of mass across **similar sequences** should be interpreted as high confidence. 

This repo contains a reimplementation of the original experiments from the paper, in the [/experiments](/experiments) folder.

You can also compute similarity-sensitive entropy (S3E) or similarity-sensitive surprisal (SSS) on your own datasets:

```
# Hyperparameter search to find the best value of alpha (described in the paper)
VALID_DATASET=s3e/data/mt/de-en/validation.js
TEST_DATASET=s3e/data/mt/de-en/test.js
python s3e/scripts/compute_uncertainty.py $VALID_DATASET deu_Latn eng_Latn output_validation.h5 --mode=validation

ALPHA=5 # Set this to the best value found in the previous command
python s3e/scripts/compute_uncertainty.py $TEST_DATASET deu_Latn eng_Latn output_test.h5 --mode=test --alpha=$ALPHA
```