from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer
#from easynmt import EasyNMT
from tqdm import tqdm
#model = EasyNMT('opus-mt')
class MarkdownDataset(Dataset):
    # train mark is taken as input - train mark contains markdown cells
    def __init__(self, df, training_corpus, model_name_or_path, total_max_len, md_max_len, fts):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        #old_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        #self.tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 50265)
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

        return ids, mask, fts, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]