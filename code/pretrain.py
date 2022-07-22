from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer
#from easynmt import EasyNMT
from tqdm import tqdm
#model = EasyNMT('opus-mt')
class PretrainDataset(Dataset):
    # train mark is taken as input - train mark contains markdown cells
    def __init__(self, df, training_corpus, model_name_or_path, total_max_len, md_max_len, fts):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.fts = fts
        

    def __getitem__(self, index):
        row = self.df.iloc[index]
        #print("row source", row.source)
        #print("code", [str(x) for x in self.fts[row.id]["codes"]])
        #text = model.translate(row.source, target_lang='en')

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
        #print(n_md, n_code)
        #print("one code cell", len(self.fts[row.id]["codes"][0]))
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]["codes"]],
            # Whether or not to encode the sequences with the special tokens relative to their model.
            add_special_tokens=True,
            # Truncate to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
            max_length=19,
            padding="max_length",
            truncation=True
        )
        #print(len(code_inputs))
#         features[idx]["total_code"] = total_code
#         features[idx]["total_md"] = total_md
#         features[idx]["codes"] = codes
        
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])
        # fts is percentage of md out of all tokens?    

        ids = inputs['input_ids']
        # numerical representations of tokens building the sequences that will be used as input by the model
        #print("fts", fts)
        #print("ids", ids)
        for x in code_inputs['input_ids']:
            #print(x)
            ids.extend(x[:-1])
        
        #print("ids", ids)
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)
        
        # https://huggingface.co/docs/transformers/glossary#attention-mask
        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len

        ##### MASKING FOR PRETRAINING
        #ids, mask, fts, torch.FloatTensor([row.pct_rank])
        labels = ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability` 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)
#         special_tokens_mask = [
#                 self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=False) for val in labels.tolist()
#         ]
#         special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

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
total_max_len=1024
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
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

# takes in df 
training_corpus = get_training_corpus()
train_ds = PretrainDataset(train_df_mark, training_corpus, model_name_or_path=model_name_or_path, md_max_len=md_max_len,
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

val_ds = PretrainDataset(val_df_mark, training_corpus, model_name_or_path=model_name_or_path, md_max_len=md_max_len,
                         total_max_len=total_max_len, fts=val_fts)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers,
                          pin_memory=False, drop_last=True, collate_fn = collate_fn_padd)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                        pin_memory=False, drop_last=False,collate_fn = collate_fn_padd)    

from transformers import LongformerModel, LongformerTokenizer, LongformerForMaskedLM
class PretrainingModel(nn.Module):
    def __init__(self, model_path, md_max_len):
        super(PretrainingModel, self).__init__()
        self.attention_window = 512
        self.md_max_len = md_max_len
        self.max_input_len = 1024
        self.max_input_len += 2
        # lengthen model
        self.model = AutoModel.from_pretrained(model_path)
        config = LongformerConfig(vocab_size = self.model.config.vocab_size, max_position_embeddings = self.model.config.max_position_embeddings)
        #config.attention_mode = 'sliding_chunks'
        longformer_model_MLM = LongformerForMaskedLM(config=config)
        longformer_model = longformer_model_MLM.longformer
        print(config)
        current_max_input_len, embed_size = self.model.embeddings.position_embeddings.weight.shape
#         print(current_max_input_len, embed_size)
        new_encoder_pos_embed = self.model.embeddings.position_embeddings.weight.new_empty(self.max_input_len, embed_size)
        print("new embed size", new_encoder_pos_embed.size())
        k = 2
        step = current_max_input_len - 2
        while k < self.max_input_len - 1:
            new_encoder_pos_embed[k:(k+step)] = self.model.embeddings.position_embeddings.weight[2:]
            k += step
        longformer_model.embeddings.position_embeddings.weight.data = new_encoder_pos_embed
        
        #Attention set up
        longformer_model.config.vocab_size = self.model.config.vocab_size
        #longformer_model.config.layer_norm_eps = self.model.config.layer_norm_eps
        longformer_model.config.attention_window = [self.attention_window] * self.model.config.num_hidden_layers
        longformer_model.config.attention_window[:4] = [32,32,64,64]
        longformer_model.config.attention_window[4:6] = [128, 128]
        longformer_model.config.attention_window[6:8] = [256,256]
        longformer_model.config.attention_window[8:10] = [512, 512]
        #print(self.model.config.num_hidden_layers)
        #print(self.model.config.attention_window)
        
        for i, layer in enumerate(self.model.encoder.layer):
            longformer_self_attn_for_codebert = LongformerSelfAttention(longformer_model.config, layer_id=i)
            longformer_self_attn_for_codebert.query = layer.attention.self.query
            longformer_self_attn_for_codebert.key = layer.attention.self.key
            longformer_self_attn_for_codebert.value = layer.attention.self.value
            
            longformer_self_attn_for_codebert.query_global = copy.deepcopy(layer.attention.self.query)
            longformer_self_attn_for_codebert.key_global = copy.deepcopy(layer.attention.self.key)
            longformer_self_attn_for_codebert.value_global = copy.deepcopy(layer.attention.self.value)
            
            longformer_model.encoder.layer[i].attention.self = longformer_self_attn_for_codebert
#             longformer_model.encoder.layer[i].attention.output.dense = layer.attention.output.dense
            
#             longformer_model.encoder.layer[i].intermediate.dense = layer.intermediate.dense
            
#             longformer_model.encoder.layer[i].output.dense = layer.output.dense

        self.model =  longformer_model_MLM


    def forward(self, input_ids, attention_mask, masked_mlm_labels):
        x = self.model(input_ids= input_ids, attention_mask = attention_mask, labels = masked_mlm_labels)
        return x


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
                loss = model(input_ids = data[0].cuda(), attention_mask = data[1].cuda(), masked_mlm_labels = data[2].cuda()).to_tuple()[0]
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
            if idx % 500000 == 0:
                torch.save(model.state_dict(), checkpoint_format.format(e=e))
                #y_val, y_pred = validate(model, val_loader)
                ##corr = (y_val != y_pred).sum()
                #false = (y_val == y_pred).sum()
                #print("Preds score", false / (corr+false))
        # objective is to learn the percentage ranking
        #y_val, y_pred = validate(model, val_loader)
        #print("Preds score", y_val - y_pred)
        torch.save(model.state_dict(), checkpoint_format.format(e=e))

    return model, y_pred


model = PretrainingModel(model_name_or_path, md_max_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[i for i in range(num_gpus)])
model = model.to(torch.device("cuda"))
model, y_pred = train(model, train_loader, val_loader, epochs=epochs)    