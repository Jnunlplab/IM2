from collections import namedtuple
#from datasets import NUFDataset          #Select your training dataset here,eg：IESDataset，
from data_utils import read_dataset
from models.class5 import NUF
import json
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args, train_ctx, train_res, valid_ctx, valid_res, annos):

    if args.metric == "NUF-CLASS":
        train_dataset = NUFDataset(train_res, annos, maxlen=args.res_token_len)
        valid_dataset = NUFDataset(valid_res, annos, maxlen=args.res_token_len)
        model = NUF(args).to(device)

    else:
        raise Exception()

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,num_workers=args.num_workers)

    trainer = pl.Trainer(max_epochs=args.max_epochs, weights_save_path=args.weight_path, nb_sanity_val_steps=5)
    trainer.fit(model, train_dataloader, valid_dataloader)
    print('[!] training complete')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IES-CLASS training script')
    parser.add_argument('--metric', type=str, default="NUF-CLASS")
    parser.add_argument('--weight-path', type=str, default='checkpoints', help='Path to directory that stores the weight')
    parser.add_argument('--pretrained-model-path', default='../../ckpt/roberta-base', help='model path of pretrained gpt2 finetuned on dataset')
    # Dataset
    parser.add_argument('--train-ctx-path', type=str, default='../../dataset/dstc10-split-by-quality-for-train/NUF/ctx.txt', help='Path to context training set')
    parser.add_argument('--train-res-path', type=str, default="../../dataset/dstc10-split-by-quality-for-train/NUF/res.txt",  help='Path to response training set')
    parser.add_argument('--valid-ctx-path', type=str, default="../../dataset/dstc10-split-by-quality-for-train/NUF/ctx.txt", help='Path to context validation set')
    parser.add_argument('--valid-res-path', type=str, default="../../dataset/dstc10-split-by-quality-for-train/NUF/res.txt",  help='Path to response validation set')
    parser.add_argument('--batch-size', type=int, default=16, help='samples per batches')
    parser.add_argument('--max-epochs', type=int, default=10, help='number of epoches to train')
    parser.add_argument('--num-workers', type=int, default=1, help='number of worker for dataset')
    parser.add_argument('--ctx-token-len', type=int, default=25, help='number of tokens for context')
    parser.add_argument('--res-token-len', type=int, default=25, help='number of tokens for response')

    # Modeling
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization')

    args = parser.parse_args()

    train_ctx = read_dataset(args.train_ctx_path) if args.train_ctx_path else None
    train_res = read_dataset(args.train_res_path)
    valid_ctx = read_dataset(args.valid_ctx_path) if args.valid_ctx_path else None
    valid_res = read_dataset(args.valid_res_path)

    labels = []
    with open("../../dataset/dstc10-split-by-quality-for-train/NUF/anno.txt") as f:
        for line in f:
            labels.append(int(line))
    train(args, train_ctx, train_res, valid_ctx, valid_res, labels)
    print("[!] done")
