import json
import os
from tqdm import tqdm
from models.abba import ABBA
import argparse
import torch
import re
from numpy import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def handle(res):

    score_list = []
    for each_res in res:
        avg_score = 0
        sub_res = re.split(r'[;,.!?]\s*', each_res)

        for idx in range(len(sub_res)-1):
            sub_pair = zip(sub_res[idx], sub_res[idx+1])
            score = model.predict(*sub_pair)
            avg_score += score
        avg_score = avg_score / len(sub_res)-1
        score_list.append(avg_score)

    return score_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='LSC')
    # parser.add_argument('--test-path', type=str, required=True, help='Path to the directory of testing set')
    parser.add_argument('--weight-path', type=str, default='./checkpoints', help='Path to directory that stores the weight')
    args = parser.parse_args()

    model = ABBA.load_from_checkpoint(checkpoint_path=args.weight_path)
    model = model.to(device)
    model.eval()

    datasets = os.listdir("../../dataset/dstc10-split-by-dialog-score")
    for each_dataset in tqdm(datasets):
        res = []
        with open("../../dataset/dstc10-split-by-dialog-score/{}/{}_all_res.txt".format(each_dataset, each_dataset),
                  encoding="utf-8") as f:
            for line in f:
                res.append(str(line.strip()))
        scores = handle(res)
        with open("nearest_turn_s/{}_score.json".format(each_dataset), "w", encoding='utf-8') as f:
            json.dump(scores, f)
        print("{}....score finished!".format(each_dataset))
