import os
import json
import logging
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
import torch.nn.functional as F
from data_utils import encode_truncate
import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ABBA(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = GPT2Tokenizer.from_pretrained(hparams.pretrained_model_path)
        self.model = GPT2Model.from_pretrained(hparams.pretrained_model_path)
        self.dropout = nn.Dropout(p=hparams.dropout, inplace=False)
        self.next_sent = nn.Linear(1024, 2)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    def forward(self, input_ids, token_type_ids=None, mask_tokens=None, pos_ids=None):
        params = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": mask_tokens,
            "position_ids": pos_ids,
        }
        last_hidden_state, _ = self.model(**params)

        mask_tokens = mask_tokens.unsqueeze(-1).repeat((1, 1, 1024))

        min_values = (torch.ones_like(mask_tokens) * -100).type(torch.FloatTensor).to(device)
        hidden_state = mask_tokens * last_hidden_state
        hidden_state = torch.where(mask_tokens != 0, hidden_state, min_values)
        hidden_state, _ = hidden_state.max(dim=1)

        hidden_state = self.dropout(hidden_state)
        next_sentence = self.next_sent(hidden_state)
        return next_sentence


    def predict(self, ctx, res):

        inputs = encode_truncate(self.tokenizer, ctx, res,
            ctx_token_len=self.hparams.ctx_token_len,
            res_token_len=self.hparams.res_token_len
        )

        with torch.no_grad():
            input_ids, token_type_ids, mask_tokens, pos_ids = [ x.unsqueeze(0).to(device) for x in inputs ]

            outputs = self(input_ids, token_type_ids=token_type_ids, mask_tokens=mask_tokens, pos_ids=pos_ids)

            outputs = F.softmax(outputs, dim=1)
            return outputs[:, 1].item()

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
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        print ("val_loss: ", avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
