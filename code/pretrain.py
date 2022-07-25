from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

class PretrainDataset(Dataset):
    # train mark is taken as input - train mark contains markdown cells
    def __init__(self, df, model_name_or_path, total_max_len, md_max_len, fts):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.fts = fts
        

    def __getitem__(self, index):
        row = self.df.iloc[index]
        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            # 64
            max_length=self.md_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        n_md = self.fts[row.id]["total_md"]
        n_code = self.fts[row.id]["total_code"]
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]["codes"]],
            # Whether or not to encode the sequences with the special tokens relative to their model.
            add_special_tokens=True,
            # Truncate to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
            max_length=19,
            padding="max_length",
            truncation=True
        )
        
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])
        # fts is percentage of md out of all tokens?    

        ids = inputs['input_ids']
        # numerical representations of tokens building the sequences that will be used as input by the model=
        for x in code_inputs['input_ids']:
            #print(x)
            ids.extend(x[:-1])
        
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)
        
        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len

        ##### MASKING FOR PRETRAINING
        labels = ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability` 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)

        #probability_matrix.masked_fill_(mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return ids, mask, labels

    def __len__(self):
        return self.df.shape[0]
    
import json
from pathlib import Path
from pathlib import Path
from datasets import load_dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import nn
from model import *
from tqdm import tqdm
import sys, os
from metrics import *
import torch
import argparse
from stlr import SlantedTriangular
data_dir = str(Path.cwd()) + '/data/'

model_name_or_path='microsoft/codebert-base'
train_mark_path=data_dir+ 'train_mark.csv'
train_features_path=data_dir+ 'train_fts.json'
val_mark_path=data_dir+ 'val_mark.csv'
val_features_path=data_dir+ 'val_fts.json'
val_path=data_dir+ 'val.csv'
checkpoint_format="./outputs/model-{e}.bin"

num_gpus=4
md_max_len=64
total_max_len=512
batch_size=4
accumulation_steps=4
epochs=5
n_workers=4

#os.mkdir("./outputs")

print("MODEL CONFIGS")

train_df_mark = pd.read_csv(train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
train_fts = json.load(open(train_features_path))
val_df_mark = pd.read_csv(val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(val_features_path))
val_df = pd.read_csv(val_path)

order_df = pd.read_csv(data_dir+"train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir + 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

raw_datasets = load_dataset("code_search_net", "python")
train_ds = PretrainDataset(train_df_mark, model_name_or_path=model_name_or_path, md_max_len=md_max_len,
                           total_max_len=total_max_len, fts=train_fts)
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    inputs = torch.nn.utils.rnn.pad_sequence([ t[0] for t in batch], batch_first=True)
    mask = torch.nn.utils.rnn.pad_sequence([ t[1] for t in batch], batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence([ t[2] for t in batch], batch_first=True)
    return inputs, mask, labels

val_ds = PretrainDataset(val_df_mark, model_name_or_path=model_name_or_path, md_max_len=md_max_len,
                         total_max_len=total_max_len, fts=val_fts)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers,
                          pin_memory=False, drop_last=True, collate_fn = collate_fn_padd)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                        pin_memory=False, drop_last=False,collate_fn = collate_fn_padd)    


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            with torch.cuda.amp.autocast():
                # inputs dim is 4
                loss, pred = model(input_ids = data[0], attention_mask = data[1], masked_mlm_labels = data[2])

            #labels.append(data[2].detach().cpu().numpy().ravel())
            #preds.append(pred.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)

def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(epochs * len(train_loader) / accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    #scheduler = SlantedTriangular(optimizer,epochs,num_steps_per_epoch=num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    
    resume_from_epoch = 0
    for try_epoch in range(epochs, 0, -1):
        if os.path.exists('./outputs/model-{epoch}.bin'.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch+1
            break
    if resume_from_epoch:
        filepath = checkpoint_format.format(e=resume_from_epoch)
        checkpoint = torch.load(checkpoint_format.format(e=try_epoch))
        model.load_state_dict(checkpoint)
        
    for e in range(resume_from_epoch, epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []

        for idx, data in enumerate(tbar):
            with torch.cuda.amp.autocast():
                loss = model(input_ids = data[0].cuda(), attention_mask = data[1].cuda(), labels = data[2].cuda()).to_tuple()[0]
                scaler.scale(loss.mean()).backward()
            if idx % accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            loss_list.append(loss.mean().detach().cpu().item())
            #preds.append(pred.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {optimizer.param_groups[0]['lr']}")
            if idx % 50000 == 0:
                torch.save(model.state_dict(), checkpoint_format.format(e=e))
                #y_val, y_pred = validate(model, val_loader)
                ##corr = (y_val != y_pred).sum()
                #false = (y_val == y_pred).sum()
                #print("Preds score", false / (corr+false))
        # objective is to learn the percentage ranking
        #y_val, y_pred = validate(model, val_loader)
        #print("Preds score", y_val - y_pred)
        torch.save(model.state_dict(), checkpoint_format.format(e=e))

    return model

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[i for i in range(num_gpus)])
model = model.to(torch.device("cuda"))
model = train(model, train_loader, val_loader, epochs=epochs)    