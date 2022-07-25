from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer
#from easynmt import EasyNMT
from tqdm import tqdm
import pandas as pd
#model = EasyNMT('opus-mt')
class MarkdownDataset(Dataset):
    # train mark is taken as input - train mark contains markdown cells
    def __init__(self, df, model_name_or_path, total_max_len, md_max_len, fts):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.df['id'] = df['id'].astype("str")
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
        #print(n_md, n_code)
        items = [str(x) for x in self.fts[row.id]["codes"]]
        code_inputs=None
        try:
            code_inputs = self.tokenizer.batch_encode_plus(
                items,
            # Whether or not to encode the sequences with the special tokens relative to their model.
                add_special_tokens=True,
                # Truncate to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
                max_length=23,
                padding="max_length",
                truncation=True
            )
        except Exception as e:
            print(e, row.id)
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[:-1])
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

        return ids, mask, fts, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]