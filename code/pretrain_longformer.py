from transformers import LongformerModel, LongformerTokenizer, LongformerForMaskedLM
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig, LongformerForMaskedLM
from pathlib import Path

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

        self.model =  longformer_model_MLM


    def forward(self, input_ids, attention_mask, masked_mlm_labels):
        x = self.model(input_ids= input_ids, attention_mask = attention_mask, labels = masked_mlm_labels)
        return x