import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig, LongformerForMaskedLM, RobertaForMaskedLM
from pathlib import Path
from pretrain_longformer import PretrainingModel

class MarkdownModel(nn.Module):
    def __init__(self, model_path, md_max_len, num_gpus, pretrained_model_path):
        super(MarkdownModel, self).__init__()
        self.attention_window = 512
        self.md_max_len = md_max_len
        self.max_input_len = 512
        self.model = AutoModel.from_pretrained(model_path)
        if pretrained_model_path:
            model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
            model = nn.DataParallel(model, device_ids=[i for i in range(num_gpus)])
            model.to("cuda")
            checkpoint = torch.load(pretrained_model_path)
            model.load_state_dict(checkpoint)
            self.model = model.module.roberta
        self.dense = nn.Linear(769, 769)
        self.dropout = nn.Dropout(0.1)
        self.top = nn.Linear(769, 1)
        
    def forward(self, ids, mask, fts):
        x = self.model(input_ids=ids, attention_mask=mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.top(x)
        return x

class LongformerModel(nn.Module):
    def __init__(self, model_path, md_max_len, using_pretrained, num_gpus):
        super(LongformerModel, self).__init__()
        self.attention_window = 512
        self.md_max_len = md_max_len
        self.max_input_len = 1024
        self.max_input_len += 2
        # lengthen model
        self.model = AutoModel.from_pretrained(model_path)
        config = LongformerConfig(vocab_size = self.model.config.vocab_size, max_position_embeddings = self.model.config.max_position_embeddings)
        longformer_model = LongformerModel(config=config)
        print(config)
        current_max_input_len, embed_size = self.model.embeddings.position_embeddings.weight.shape
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
        for i, layer in enumerate(self.model.encoder.layer):
            longformer_self_attn_for_codebert = LongformerSelfAttention(longformer_model.config, layer_id=i)
            longformer_self_attn_for_codebert.query = layer.attention.self.query
            longformer_self_attn_for_codebert.key = layer.attention.self.key
            longformer_self_attn_for_codebert.value = layer.attention.self.value
            
            longformer_self_attn_for_codebert.query_global = copy.deepcopy(layer.attention.self.query)
            longformer_self_attn_for_codebert.key_global = copy.deepcopy(layer.attention.self.key)
            longformer_self_attn_for_codebert.value_global = copy.deepcopy(layer.attention.self.value)
            
            longformer_model.encoder.layer[i].attention.self = longformer_self_attn_for_codebert
        self.model = longformer_model
        if using_pretrained:
            model = PretrainingModel(model_path, md_max_len)
            model = nn.DataParallel(model, device_ids=[i for i in range(num_gpus)])
            model.to("cuda")
            checkpoint = torch.load(str(Path.cwd())+"/outputs/model-0.bin")
            model.load_state_dict(checkpoint)
            self.model = model.module.model.longformer
        
        self.top = nn.Linear(769, 1)
    def forward(self, ids, mask, fts):
        global_attention_mask = torch.zeros_like(ids)
        global_attention_mask[:, :64] = 1
        x = self.model(input_ids=ids, attention_mask=mask, global_attention_mask=None)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        x = self.top(x)
        return x
        
