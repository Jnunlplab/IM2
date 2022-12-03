import codecs
import json
import os
import torch
import numpy as np
import math
import argparse
from tqdm import tqdm
vocab = {}
vocab_total_num = 0
def vocab_generate(candsf, ngram):
    with open(candsf, encoding="utf-8") as f:
        cands = [line.strip() for line in f]
    for s in cands:
        if not s:
            continue
        ws = s.split(' ')
        if len(ws) <= ngram:
            k = ' '.join(ws)
            if not k in vocab:
                vocab[k] = 1
            else:
                vocab[k] = vocab[k] + 1
        else:
            for i in range(len(ws) - ngram + 1):
                k = ' '.join(ws[i:i + ngram])
                if not k in vocab:
                    vocab[k] = 1
                else:
                    vocab[k] = vocab[k] + 1
    global vocab_total_num
    vocab_total_num = sum([v for v in vocab.values()])
    print(max(zip(vocab.values(), vocab.keys())))

def diversity_entropy_score(utter, ngram):
    ws = utter.split(' ')
    now_vocab = {}
    if len(ws) <= ngram:
        k = ' '.join(ws)
        if not k in  now_vocab:
            now_vocab[k] = 1
        else:
            now_vocab[k] = now_vocab[k] + 1
    else:
        for i in range(len(ws) - ngram + 1):
            k = ' '.join(ws[i:i + ngram])
            if not k in now_vocab:
                now_vocab[k] = 1
            else:
                now_vocab[k] = now_vocab[k] + 1
    entropy = 0
    for w in now_vocab.keys():
        v = vocab[w]
        entropy += -(v / vocab_total_num) * np.log((v / vocab_total_num))
    return entropy/len(now_vocab)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='N-gram for each m lines')
    args_parser.add_argument('--num_line', type=int, default=5, help='Number of lines for each evaluation')
    args_parser.add_argument('--ngram', type=int, default=2, help='N-gram')
    args_parser.add_argument('--ngram_choice', choices=['nline_ngram_entropy', 'ngram_entropy'],
                             default='nline_ngram_entropy',
                             help='Choice of N-gram')
    args_parser.add_argument('--cands_file', default='data/dailydialog_test_res.txt', type=str,
                             help='cands_file')
    args_parser.add_argument('--output_file', default='data/result.txt', type=str,
                             help='output_file')
    args = args_parser.parse_args()

    datasets = os.listdir("../../dataset/dstc10-split-by-dialog-score")
    for each_dataset in tqdm(datasets):
        vocab.clear()
        vocab_total_num = 0
        diversity_score = []
        filepath_ctx = "../../dataset/dstc10-split-by-dialog-score/{}/{}_all_ctx.txt".format(each_dataset, each_dataset)
        vocab_generate(filepath_ctx, args.ngram)
        filepath_res = "../../dataset/dstc10-split-by-dialog-score/{}/{}_all_res.txt".format(each_dataset, each_dataset)
        vocab_generate(filepath_res, args.ngram)

        with open(filepath_res, encoding="utf-8") as f:
            for line in f:
                now_entropy_score = diversity_entropy_score(line.strip(), args.ngram)
                diversity_score.append(now_entropy_score)
        norm_diversity_score = []

        for x in diversity_score:
            x = float(x - min(diversity_score)) / (max(diversity_score) - min(diversity_score))
            norm_diversity_score.append(x)

        with open("diversity_s/{}_score.json".format(each_dataset), "w", encoding='utf-8') as f:
            json.dump(norm_diversity_score, f)
        print("{}....score finished!".format(each_dataset))


