# IM2
This repository contains the source code of our emnlp 2022 paper. IM2: an Interpretable and Multi-category Integrated Metric Framework for Automatic Dialogue Evaluation
![image](https://user-images.githubusercontent.com/116079501/196417237-3dc638c0-193e-415e-9701-82c0f9bbd17b.png) 

Create virtural environment:
## conda create -n im2 python=3.6
## source activate im2
Install the required packages:
## pip install -r requirements.txt

# Processed the data
we provide the processed data,train/valid/test datasets,all processed dstc10 datasets.

# ckpt
the checkpoint is provided，contains fne-tuned dialogpt, Roberts, and ab-ba,ab-bc for IM2,etc..

# get score
You can either use your own data or the data that we have processed, use all sub-metric to get the score, and then  get the combined IM2 score.



