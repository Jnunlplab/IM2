import os
import json
import logging
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, BertModel, BertTokenizer
import torch.nn.functional as F
from data_utils import encode_truncate
import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NUF(pl.LightningModule):   # NUF or IES model
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #self.tokenizer = RobertaTokenizer.from_pretrained(hparams.pretrained_model_path)
        #self.bert = RobertaModel.from_pretrained(hparams.pretrained_model_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        num_classes = 5
        self.mlp_hidden_0 = nn.Linear(768, 64, bias=True)
        self.mlp_hidden_1 = nn.Linear(64, 32, bias=True)
        self.mlp_hidden_2 = nn.Linear(32,  16, bias=True)
        self.mlp_out = nn.Linear(16, num_classes, bias=True)
        self.dropout = nn.Dropout(p=hparams.dropout, inplace=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    def forward(self, input_ids, token_type_ids=None, mask_tokens=None, pos_ids=None):

        params = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": mask_tokens,
            "position_ids": pos_ids,
        }

        # run BERT with input
        last_hidden_state, _ = self.bert(**params)

        # max-pooling
        mask_tokens = mask_tokens.unsqueeze(-1).repeat((1, 1, 768))
        min_values = (torch.ones_like(mask_tokens) * -100).type(torch.FloatTensor).to(device)
        hidden_state = mask_tokens * last_hidden_state
        hidden_state = torch.where(mask_tokens != 0, hidden_state, min_values)
        hidden_state, _ = hidden_state.max(dim=1)

        hidden_state = self.dropout(hidden_state)

        hidden_0 = self.mlp_hidden_0(hidden_state)
        hidden_1 = self.mlp_hidden_1(hidden_0)
        hidden_2 = self.mlp_hidden_2(hidden_1)
        out = self.mlp_out(hidden_2)

        return out

    def predict(self, ctx, res):

        inputs = encode_truncate(self.tokenizer, ctx, res,
            ctx_token_len=self.hparams.ctx_token_len,
            res_token_len=self.hparams.res_token_len
        )

        with torch.no_grad():
            input_ids, token_type_ids, mask_tokens, pos_ids = [x.unsqueeze(0).to(device) for x in inputs ]
            #outputs = self(input_ids, token_type_ids=token_type_ids, mask_tokens=mask_tokens, pos_ids=pos_ids)
            outputs = self(input_ids, mask_tokens=mask_tokens, pos_ids=pos_ids)

            outputs = F.softmax(outputs, dim=1)

            p1 = outputs[:, 0]
            p2 = outputs[:, 1]
            p3 = outputs[:, 2]
            p4 = outputs[:, 3]
            p5 = outputs[:, 4]
            score = 1 * p1.item() + 2 * p2.item() + 3 * p3.item() + 4 * p4.item() + 5 * p5.item()
            #return outputs[:, 1].item()
            return score


    def training_step(self, batch, batch_nb):
        input_ids, token_type_ids, mask_tokens, pos_ids, label = [ x.to(device) for x in batch ]

        params = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "mask_tokens": mask_tokens,
            "pos_ids": pos_ids,
        }
        outputs = self(**params)
        loss = self.criterion(outputs, label)
        return { 'loss': loss }

    def validation_step(self, batch, batch_nb):
        output = self.training_step(batch, batch_nb)
        return {'val_loss': output['loss'] }

    def validation_epoch_end(self, outputs):
        print("hello!")
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        print ("val_loss: ", avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
