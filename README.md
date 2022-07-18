# Introduction
The objective for the AI4Code Competition is to train a model that effectively orders shuffled markdown cells in a kaggle notebook where its code cells are fixed. Past Kaggle notebooks created are given as training data and the final submission on evaluated on newly created Kaggle Notebooks. The loss is measured by kendall tau rank correlation coefficient.

# Baseline Solution
This repository is a fork of an early open sourced solution by @suicao which is based on Amet Erdem's [baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline). While the Amet Erdem's solution predicts cell position only with the markdown, suicao's solution uses [codebert pretrained weights](https://huggingface.co/microsoft/codebert-base) and a sample of up to 20 code cells from the notebook as global context to help determine the percentage rank of the markdown cell's position. 

Below is an example input for the model:
```<s> Markdown content <s> Code content 1 <s> Code content 2 <s> ... <s> Code content 20 <s> ```

As for specific implementation details, the model takes in 512 tokens - 64 tokens for markdown and 20 23-token length code cells. The tokens are encoded with codebert, its output is concatenated with the float value representation of the percentage of markdown cells out of all cells in the notebook. 

# Code Execution

### Preprocessing
To extract features for training, including the markdown-only dataframes and sampling the code cells needed for each note book, simply run:

```$ python preprocess.py```

Your outputs will be in the ```./data``` folder:
```
project
│   train_mark.csv
│   train_fts.json   
|   train.csv
│   val_mark.csv
│   val_fts.json
│   val.csv
```

###  Training
From the original code, I added a parameter for number of GPUs. Training per epoch took around 3 hours on 4 V100 GPUs.
```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 8 --accumulation_steps 4 --epochs 5 --n_workers 4 --num_gpus

The validation scores should read 0.84+ after 3 epochs, and also correlates well with the public LB.

### Inference
Original notebook: https://www.kaggle.com/code/suicaokhoailang/stronger-baseline-with-code-cells

# Research
I had to research several topics when preparing for this competition, many of them either seemed infeasible or yielded negative results. I thought of potential reasons why they might not have been successful but I am new to this domain so I am not fully certain.

## Sentence Ordering
The objective of this competition is to order markdown cells which is effectively natural language. Naturally, the first domain I researched was sentence ordering. [BERT4SO](https://arxiv.org/abs/2103.13584) is a paper which proposes sentence ordering with an architecture below.  
## NL-PL Language Models
There are quite a few NL-PL models that have been proposed by researchers. Personally, I have only tried using codebert and graphcodebert
## Longformer
A significant limitation for the challenge is that the input size for the model is only 512 tokens when the markdown cell and code cells in a single notebook is longer than 512 tokens. Just taking the product of the mean code cell count and mean code cell length is around 600 tokens.

One 

I read through the approach for using hiearchical transformers for long document summarization and considered using frozen codebert and roberta to encode segments and adding a transformer layer on top but it seemed unfeasible as long document summarization uses average pooling at the end so the model but I would need to encode each markdown cell 
