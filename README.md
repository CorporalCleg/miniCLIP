# miniCLIP
This repo is a small version of CLIP model. Here we used ResNet50 and DistilledBERT as vision and text encoder. As dataset [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) was used.
## Installation 

```shell
conda env create -f environment.yml
conda activate clip-env
```

## Reproducing results

### Training scripts

To train  model

```shell
cd src
python train_eval.py
```
### Zero-shot classification

To make a zero-shot you will need to download [checkpoint]() and put into ```ckpt/``` and [zero-shot data]() to ```zero shot``

```shell
cd src/
python zero_shot.py
```

![](extra/pred_vs_gt.png)