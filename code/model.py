import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig


class MarkdownModel(nn.Module):
    def __init__(self, model_path, md_max_len):
        super(MarkdownModel, self).__init__()
        self.attention_window = 512
        self.md_max_len = md_max_len
        self.max_input_len = 1024
        
        self.model = AutoModel.from_pretrained(model_path)
        #print(self.model.encoder)
        config = self.model.config
        current_max_pos, embed_size = self.model.embeddings.position_embeddings.weight.shape
        self.max_input_len += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
        config.max_position_embeddings = self.max_input_len
        print(self.max_input_len, current_max_pos)
        assert self.max_input_len >= current_max_pos
        new_pos_embed = self.model.embeddings.position_embeddings.weight.new_empty(self.max_input_len, embed_size)
        k = 2
        step = current_max_pos - 2
        while k < self.max_input_len - 1:
            print(step, k)
            new_pos_embed[k:(k + step)] = self.model.embeddings.position_embeddings.weight[2:]
            k += step
        self.model.embeddings.position_embeddings.weight.data = new_pos_embed
        self.max_input_len, embed_size)print(self.model.embeddings.weight)
        print(self.model.embeddings)
        self.model.embeddings.position_ids.data = torch.tensor([i for i in range(self.max_input_len)]).reshape(1, self.max_input_len)

        # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
        config.attention_window = [self.attention_window] * self.model.config.num_hidden_layers
        for i, layer in enumerate(self.model.encoder.layer):
            longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
            #print(longformer_self_attn.query)
            longformer_self_attn.query = layer.attention.self.query            
            longformer_self_attn.key = layer.attention.self.key
            longformer_self_attn.value = layer.attention.self.value

            longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
            longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
            longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

            layer.attention.self = longformer_self_attn
        print(config)
        print(self.model.embeddings.token_type_ids)
        
        print("embeddings", self.model.embeddings)
        #print(self.model.encoder)
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        #global_attention_mask = torch.zeros_like(ids)
        #print("ids", ids.size())
        #global_attention_mask[:, :] = 1
        print(ids.size())
        print(mask.size())
        x = self.model(input_ids=ids, attention_mask=mask)[0]
        #print("fts", fts)
        #print("x size", x.size())
        #print("values", torch.mean(x[:, 0:64, :]), fts)
        x = torch.cat((torch.mean(x[:, 0:64, :], axis=1), fts), 1)
        #print("/n", x.size())
        x = self.top(x)
        return x
        
