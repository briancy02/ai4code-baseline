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
        #self.max_input_len = 16384
        #self.max_input_len += 2
        self.attention_window = 512
        self.md_max_len = md_max_len
        # lengthen model
        self.model = AutoModel.from_pretrained(model_path)
        longformer_model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        current_max_input_len, embed_size = self.model.embeddings.position_embeddings.weight.shape
#         print(current_max_input_len, embed_size)
#         new_encoder_pos_embed = self.model.embeddings.position_embeddings.weight.new_empty(self.max_input_len, embed_size)
#         k = 2
#         step = current_max_input_len - 2
#         while k < self.max_input_len - 1:
#             new_encoder_pos_embed[k:(k+step)] = self.model.embeddings.position_embeddings.weight[2:]
#             k += step
#         self.model.embeddings.position_embeddings.weight.data = new_encoder_pos_embed
#         print(self.model.embeddings.position_embeddings.weight.shape)
        
        
        #Attention set up
        self.model.config.attention_window = [self.attention_window] * self.model.config.num_hidden_layers
        #print(self.model.config.attention_window)
        self.model.config.attention_probs_dropout_prob = 0
        
        for i, layer in enumerate(self.model.encoder.layer):
            longformer_self_attn_for_codebert = LongformerSelfAttention(self.model.config, layer_id=i)
            longformer_self_attn_for_codebert.query = layer.attention.self.query
            longformer_self_attn_for_codebert.key = layer.attention.self.key
            longformer_self_attn_for_codebert.value = layer.attention.self.value
            
            longformer_self_attn_for_codebert.query_global = copy.deepcopy(layer.attention.self.query)
            longformer_self_attn_for_codebert.key_global = copy.deepcopy(layer.attention.self.key)
            longformer_self_attn_for_codebert.value_global = copy.deepcopy(layer.attention.self.value)
            
            longformer_model.encoder.layer[i].attention.self = longformer_self_attn_for_codebert
        self.model = longformer_model
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        global_attention_mask = torch.zeros_like(ids)
        global_attention_mask[:self.md_max_len] = 0
        x = self.model(input_ids=ids, attention_mask=mask, global_attention_mask=global_attention_mask)[0]
        #print("fts", fts)
        x = torch.cat((x[:, 0, :], fts), 1)
        #print("/n", x.size())
        x = self.top(x)
        return x
        
