# Occupational Depictions and Realities in China

## Goals
- Use word embedding techniques to get relationships between words in text
- Use these relationships to determine how the “stereotype” (in terms of gender, education level, “prestige”, etc.) of occupations has changed over the past few decades

## Dataset
- Renmin Ribao (People’s Daily) newspaper
- 1946-2003
- Each row is an article with 4 columns: date, page, section, and content
- Focus on content

## Get word embeddings
### Hyperparameters:
- Each word represented by vector of 300 dimensions
- Words have to appear at least 100 times in a decade to be included in training
- Trainings are based on decades (1960-1969 corpus pooled together)
- 5 epochs are used. Current decade’s embedding is trained on previous decade’s embedding
- 5 words are negatively sampled in each update

### 3 models:
1. No downsampling
2. Stop words are downsampled
3. Stop words are downsampled and punctuation is removed

To get word embeddings, use command
```
python w2v_dec.py
```

To write word embeddings to text file, use command
```
python write_embeddings.py
```

## Evaluate word embeddings
### Metrics:
1. Similarity score: Measure of how similar two words are
2. Analogy problem: “A is to B as C is to D”

### Checks:
To check word embeddings, use commands
```
python similarity.py
python analogy.py
```

### Tests:
To get similarity correlation coefficients, use command
```
python /COS960-master/correlation_calcu.py {WORD_EMBEDDING_TXTFILE} {OUTPUT_TXTFILE}
```

To get analogy accuracies, use command
```
python ana_eval_dense.py -v {WORD_EMBEDDING_TXTFILE} -a /Chinese-Word-Vectors-master/testsets/CA8/morphological.txt
python ana_eval_dense.py -v {WORD_EMBEDDING_TXTFILE} -a /Chinese-Word-Vectors-master/testsets/CA8/semantic.txt
```

For more details, see citations.

## Explore trends
Get distances between occupations and "stereotypes"

### Word lists:
- Stereotype word lists: 14 lists of words relating to “female,” “male,” “prestige,” “common,” “education,” “uneducation,” “affluent,” “poor,” “good,” “bad,” “strong,” “weak,” “active,” “passive”
- Occupation list: List of 315 jobs

## Citations
```
@article{huang2019COS960,
Author = {Junjie Huang and Fanchao Qi and Chenghao Yang and Zhiyuan Liu and Maosong Sun},
Title = {{COS960: A Chinese Word Similarity Dataset of 960 Word Pairs}},
journal={arXiv preprint arXiv:1906.00247},
Year = {2019},
}

@InProceedings{P18-2023,
  author =  "Li, Shen
    and Zhao, Zhe
    and Hu, Renfen
    and Li, Wensi
    and Liu, Tao
    and Du, Xiaoyong",
  title =   "Analogical Reasoning on Chinese Morphological and Semantic Relations",
  booktitle =   "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
  year =  "2018",
  publisher =   "Association for Computational Linguistics",
  pages =   "138--143",
  location =  "Melbourne, Australia",
  url =   "http://aclweb.org/anthology/P18-2023"
}
```
