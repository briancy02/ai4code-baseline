import json
from pathlib import Path
from dataset import *
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
parser.add_argument('--data_dir', type=str, default="processed_dataset_1/")

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
parser.add_argument('--train_mark_path', type=str, default='train_mark.csv')
parser.add_argument('--train_features_path', type=str, default='train_fts.json')
parser.add_argument('--val_mark_path', type=str, default='val_mark.csv')
parser.add_argument('--val_features_path', type=str, default='val_fts.json')
parser.add_argument('--val_path', type=str, default='val.csv')
parser.add_argument('--orders_data_path', type=str, default='train_orders.csv')
parser.add_argument('--checkpoint_format', type=str, default="./outputs/codebert_new_data/model-{e}.bin")


parser.add_argument('--num_gpus', type=int, default=4)
parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)


args = parser.parse_args()
data_dir = str(Path.cwd()) + '/data/' + args.data_dir

#os.mkdir("./outputs")

print("MODEL CONFIGS")

if args.data_dir == "processed_dataset_2/":
    train_df_mark = pd.read_csv(data_dir+args.train_mark_path +).dropna().reset_index(drop=True)
    val_df_mark = pd.read_csv(data_dir+args.val_mark_path).dropna().reset_index(drop=True)
    df_orders = pd.read_csv(
        data_dir + 'data.csv',
        index_col='id',
        squeeze=True,
    )
    df_orders = df_orders[df_orders['cell_type']=='markdown'].drop(columns = ['source','cell_type', 'rank', 'pct_rank'])
else:    
    train_df_mark = pd.read_csv(data_dir+args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
    val_df_mark = pd.read_csv(data_dir+args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
    df_orders = pd.read_csv(
        data_dir + 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split() 
train_fts = json.load(data_dir+open(args.train_features_path))
val_fts = json.load(data_dir+open(args.val_features_path))
val_df = pd.read_csv(data_dir+args.val_path)






raw_datasets = load_dataset("code_search_net", "python")
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

# takes in df 
training_corpus = get_training_corpus()
train_ds = MarkdownDataset(train_df_mark, training_corpus, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                           total_max_len=args.total_max_len, fts=train_fts)
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    inputs = torch.nn.utils.rnn.pad_sequence([ t[0] for t in batch], batch_first=True)
    masks = torch.nn.utils.rnn.pad_sequence([ t[1] for t in batch], batch_first=True)
    fts = torch.nn.utils.rnn.pad_sequence([ t[2] for t in batch], batch_first=True)
    ranks = torch.nn.utils.rnn.pad_sequence([ t[3] for t in batch], batch_first=True)
    
    return inputs, masks, fts, ranks

val_ds = MarkdownDataset(val_df_mark, training_corpus, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                          pin_memory=False, drop_last=True, collate_fn = collate_fn_padd)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False,collate_fn = collate_fn_padd)

    

def read_data(data):
    # seperation of training data and label
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                # inputs dim is 4
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)

# Model takes one markdown cell and sampled code as a single datapoint, learns to generate correct percent rank

def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    #scheduler = SlantedTriangular(optimizer,epochs,num_steps_per_epoch=num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    
    resume_from_epoch = 0
    for try_epoch in range(epochs, 0, -1):
        if os.path.exists('./outputs/codebert_new_data/model-{epoch}.bin'.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch+1
            break
    if resume_from_epoch:
        filepath = args.checkpoint_format.format(e=resume_from_epoch)
        checkpoint = torch.load(args.checkpoint_format.format(e=try_epoch))
        model.load_state_dict(checkpoint)
        
    for e in range(resume_from_epoch, epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {optimizer.param_groups[0]['lr']}")
        
        # objective is to learn the percentage ranking
        y_val, y_pred = validate(model, val_loader)
        # display rankings in percentage
        val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
        #print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
        torch.save(model.state_dict(), args.checkpoint_format.format(e=e))
        print("Preds score", kendall_tau(df_orders[df_orders.id==y_dummy.index], y_dummy))
        

    return model, y_pred


model = MarkdownModel(args.model_name_or_path, args.md_max_len, using_pretrained=True, num_gpus=args.num_gpus)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
model = model.to(device)
model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs)
