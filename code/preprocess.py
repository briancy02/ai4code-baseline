import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import os

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
#nltk.download('stopwords')
sw_nltk = stopwords.words('english')
print(type(sw_nltk))
print('me' in sw_nltk)
data_dir = str(Path.cwd()) + '/data/'
# if not os.path.exists("./data"):
#     os.mkdir("./data")
print(data_dir)

def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
            .assign(id=path.stem)
            .rename_axis('cell_id')
    )


paths_train = list(Path(data_dir + 'train').glob('*.json'))
notebooks_train = [
    read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
]
df = (
    pd.concat(notebooks_train)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
)
# l = [range(10000, 20000)]
# import pickle
# file = open("df_preprocess.pkl",'rb')
# df = pickle.load(file)

#df.drop(df.index[l], inplace=True)

df_orders = pd.read_csv(
    data_dir+'processed_dataset_1/train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()  # Split the string representation of cell_ids into a list


def get_ranks(base, derived):
    if not isinstance(base, list):
        base = []
        derived = []
    return [base.index(d) for d in derived]


df_orders_ = df_orders.to_frame().join(
    df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
    how='right',
)

ranks = {}
for id_, cell_order, cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}
df_ranks = (
    pd.DataFrame
        .from_dict(ranks, orient='index')
        .rename_axis('id')
        .apply(pd.Series.explode)
        .set_index('cell_id', append=True)
)

df_ancestors = pd.read_csv(data_dir + 'train_ancestors.csv', index_col='id')
df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
# count number of cells in notebook?
df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def read_notebook(path):
    
    return pd.read_json(path).assign(id = path.stem).rename_axis('cell_id')




def preprocess_text(document, source_type):
#     document = re.sub(r'\n\s+', '\n', document)
    # change #:
    if source_type=='markdown':
        document = re.sub(r'^#####', 'header5 ', document)
        document = re.sub(r'^####', 'header4 ', document)
        document = re.sub(r'^###', 'header3 ', document)
        document = re.sub(r'^##', 'header2 ', document)
        document = re.sub(r'^#', 'header1 ', document)
    elif source_type=='code':
        document = re.sub(r'#', 'zhushi ', document)
        document = re.sub(r'\n\'\'\'\n',' zhushi2 ', document)


#     # change \n into huanhang
#     document = re.sub(r'\n+', ' huanhang ', document)

    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
  
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    tokens = document.split()
    if source_type=='markdown':
        tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in sw_nltk]  #remove stopwords

    preprocessed_text = ' '.join(tokens)

#     preprocessed_text = re.sub(r'huanhang', '\n', preprocessed_text)
    if source_type=='code':
        preprocessed_text = re.sub(r'zhushi', '#', preprocessed_text)

    return preprocessed_text

    
def preprocess_df(df):
    """
    This function is for processing sorce of notebook
    returns preprocessed dataframe
    """
    return [preprocess_text(message, source_type) for message, source_type in tqdm(zip(df.source, df.cell_type))]



from sklearn.model_selection import GroupShuffleSplit

NVALID = 0.1  # size of validation set
splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
train_df = df.loc[train_ind].reset_index(drop=True)
train_df.source = preprocess_df(train_df)
val_df = df.loc[val_ind].reset_index(drop=True)
val_df.source = preprocess_df(val_df)



# Base markdown dataframes
train_df_mark = train_df[train_df["cell_type"] == "markdown"].reset_index(drop=True)
val_df_mark = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)
train_df_mark.to_csv("./data/processed_dataset_1/train_mark.csv", index=False)
val_df_mark.to_csv("./data/processed_dataset_1/val_mark.csv", index=False)
val_df.to_csv("./data/processed_dataset_1/val.csv", index=False)
train_df.to_csv("./data/processed_dataset_1/train.csv", index=False)




# Additional code cells
def clean_code(cell):
    return str(cell).replace("\\n", "\n")


def sample_cells(cells, n):
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells):
        return [cell[:200] for cell in cells]
    else:
        results = []
        # select cells sparsely based on the n value
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results


def get_features(df):
    features = dict()
    df = df.sort_values("rank").reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby("id")):
        features[idx] = dict()
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        
        codes = sample_cells(code_sub_df.source.values, 40)
        if len(codes)==0:
            codes = ["/n"]
        features[idx]["total_code"] = total_code
        features[idx]["total_md"] = total_md
       
        features[idx]["codes"] = codes
    return features

val_fts = get_features(val_df)
json.dump(val_fts, open("./data/processed_dataset_1/val_fts.json","wt"))
train_fts = get_features(train_df)
json.dump(train_fts, open("./data/processed_dataset_1/train_fts.json","wt"))
