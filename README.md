# IM2
This repository contains the source code of our emnlp 2022 paper. IM2: an Interpretable and Multi-category Integrated Metric Framework for Automatic Dialogue Evaluation
![image](https://user-images.githubusercontent.com/116079501/196417237-3dc638c0-193e-415e-9701-82c0f9bbd17b.png) 

# Environment
Create virtural environment:<br>
```
conda create -n im2 python=3.6
source activate im2
```
Install the required packages:<br>
```
    pip install -r requirements.txt
```

# Processed the data
we provide the processed data,train/valid/test datasets,all processed dstc10 datasets.<br>
```
   datc10 data : ./dstc10-split-by-dialog-score
   ab-ac and ab-ba train data : ./dailydialog
   test data : /test_data_anno 
```
# ckpt
the checkpoint is provided，contains fne-tuned dialogpt, Roberts, and ab-ba,ab-bc for IM2,etc..<br>
```
    cd ./ckpt 
```
You could download it and unzip <br>

# get score
You can either use your own data or the data that we have processed, use all sub-metric to get the score, and then  get the combined IM2 score.<br>
use each model and loading its corresponding ckpt,get the sub-metric score <br>
```
   NUF = w1 ∗ LSC + w2 ∗ V UP + w3 ∗ 5-NUF 
   CR = w4 ∗ GRADE + w5 ∗ AB-AC +w6 ∗ AB-BA 
   IES = w7 ∗ Dist-n +w8 ∗ D-MLM +w9 ∗ 5-IES 
   IM2 = α1 ∗ NUF + α2 ∗ CR + α3 ∗ IES 
```
  



