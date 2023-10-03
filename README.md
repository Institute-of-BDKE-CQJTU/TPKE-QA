Code and datasets of our paper "TPKE-QA: A Gapless Few-shot Extractive Question Answering Approach via Task-aware Post-training and Knowledge Enhancement".

Our post-training code is based on PyTorch(1.8.1) and Transformers(4.21.0), and fine-tuning code is identical with [splinter](https://github.com/oriram/splinter) in requirements.

### Data

#### Downloading few-Shot EQA data
```shell
curl -L https://www.dropbox.com/sh/pfg8j6yfpjltwdx/AAC8Oky0w8ZS-S3S5zSSAuQma?dl=1 > mrqa-few-shot.zip
unzip mrqa-few-shot.zip -d ./data/mrqa-few-shot
```

#### Downloading openwebtext and ccnews
Our post-training data is based on openwebtext and ccnews, openwebtext corpus can be downloaded in https://skylion007.github.io/OpenWebTextCorpus/, ccnews can be downloaded in https://storage.googleapis.com/huggingface-nlp/datasets/cc_news/cc_news.tar.gz

#### Downloading wikidata5m
The wikidata5m can be downloaded in https://deepgraphlearning.github.io/project/wikidata5m#data

### Task-aware Post-training

#### Downloading splinter model
```shell
curl -L https://www.dropbox.com/sh/h63xx2l2fjq8bsz/AAC5_Z_F2zBkJgX87i3IlvGca?dl=1 > splinter.zip
unzip splinter.zip -d ./model/splinter 
```
#### Making post-traing data

Before data construction, we first get origin text from openwebtext and ccnews, and the paragraph with less than 300 tokens in length are discarded.

```shell
python post-training/make_post-training_data.py --dataset=ccnews
```

#### Post-training 
```shell
bash post-training/run_post-training.sh
```

### Knowledge-enhanced Fine-tuning

#### Proprecessing the few-shot EQA data

```shell
bash fine-tuning/map_mrqa_data.sh
```

#### NER & EL in few-shot EQA data

```shell
python fine-tuning/get_total_entity_by_spacy.py
```

#### Fine-tuning

```shell
bash fine-tuning/run_fine-tuning.sh
```