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
        self.max_input_len = 2048
        self.max_input_len += 2
        # lengthen model
        self.model = AutoModel.from_pretrained(model_path)
        config = LongformerConfig(vocab_size = self.model.config.vocab_size, max_position_embeddings = self.model.config.max_position_embeddings)
        #config.attention_mode = 'sliding_chunks'
        longformer_model = LongformerModel(config=config)
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
        longformer_model.config.layer_norm_eps = self.model.config.layer_norm_eps
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

#         longformer_model.pooler.dense = self.model.pooler.dense    #print(longformer_model.encoder.layer[i])
        self.model = longformer_model
        print(self.model)
        #print(self.model)
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        global_attention_mask = torch.zeros_like(ids)
        #print("ids", ids.size())
        global_attention_mask[:, :64] = 1
        x = self.model(input_ids=ids, attention_mask=mask, global_attention_mask=None)[0]
        #print("fts", fts)
        #print("x size", x.size())
        #print("values", torch.mean(x[:, 0:64, :]), fts)
        x = torch.cat((x[:, 0, :], fts), 1)
        #print("/n", x.size())
        x = self.top(x)
        return x
        
