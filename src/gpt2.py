"""
GPT-2 Implementation from Scratch

This module implements the GPT-2 transformer architecture using only PyTorch.
No HuggingFace dependencies are allowed in this file.
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GPT2Config:
    """Configuration class for GPT-2 small model."""
    # Total number of tokens in the vocabulary.
    # Note we use the same vocabulary and tokenizer as OpenAI GPT-2.
    vocab_size: int = 50257
    
    # The maximum context window length is 1024 tokens for GPT-2.
    max_ctx_len: int = 1024
    
    # The model dimension (hidden size) for GPT-2 Small is 768.
    d_model: int = 768
    
    # The dimension of each attention head is d_model / n_head = 768 / 12 = 64.
    d_head: int = 64
    
    # The intermediate dimension of the MLP in GPT-2 Small is 4 times the model dimension.
    # 4 * 768 = 3072
    d_mlp_intermediate: int = 3072
    
    # GPT-2 Small has 12 transformer blocks.
    n_layer: int = 12
    
    # GPT-2 Small has 12 attention heads per transformer block.
    n_head: int = 12
    
    # Total number of label classes for our classification dataset.
    num_labels: int = 20


@dataclass
class CausalLMOutput:
    """Output class for causal language modeling. Contains the logits for all input tokens."""
    logits: Tensor


@dataclass
class ModelOutput:
    """Output class for generation. Contains sequences of input and generated token IDs."""
    sequences: Tensor


@dataclass
class SequenceClassifierOutput:
    """Output class for sequence classification. Contains the logits for each label class."""
    logits: Tensor


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.d_model, config.d_mlp_intermediate)
        self.c_proj = nn.Linear(config.d_mlp_intermediate, config.d_model)
        self.act  = nn.GELU(approximate='tanh')
        self.drop = nn.Dropout(p=0.1)
 
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.drop(x) # dropout here
        return x
 
 
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head  = config.n_head
        self.d_head  = config.d_head
        self.d_model = config.d_model
 
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.drop   = nn.Dropout(p=0.1)

    def forward(self, x, past_kv=None):
        B, T, C = x.shape 

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)

        # Scaled dot product attention
        # is_causal must be False if we are using a KV cache because 
        # the mask is implicit in the length of the cache vs query
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.1 if self.training else 0.0,
            is_causal=(past_kv is None and T > 1) 
        )

        # Crucial: Use T (current input length), not the concatenated length
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(attn_out)), new_kv
 
    
class TransformerBlock(nn.Module):
    """
    This is a single Transformer Block. Each of it contains a self-attention module and a MLP module.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1  = nn.LayerNorm(config.d_model) # first layer norm
        self.attn  = CausalSelfAttention(config) # multi-head masked attention
        self.ln_2  = nn.LayerNorm(config.d_model) # second layer norm
        self.mlp   = MLP(config) # MLP
 
    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out # residual after attention and dropout
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv

class GPT2LMHeadModel(nn.Module):
    """
    GPT-2 Language Model with a language modeling head.
    This corresponds to HF's GPT2LMHeadModel.
    """

    def __init__(self, config: GPT2Config = GPT2Config(), bin_path: Optional[str] = None):
        """
        Initialize GPT-2 Language Model.
        
        Args:
            config: GPT2Config object containing model configurations.
            bin_path: Path to the pytorch_model.bin file. If empty or None, 
                      weights will not be loaded from file.
        """
        super().__init__()
        
        # TODO: define and initialize the GPT-2 model architecture here. 
        # If the `bin_path` argument is provided, 
        # load the model weights from the specified file path.
        # If `bin_path` is empty or None, do not load any weights, 
        # and initialize the model with random weights.

        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model) # vocab_size, hidden_dim --> (50257,768)
        self.wpe   = nn.Embedding(config.max_ctx_len, config.d_model) #  max seq len , hidden dim --> (1024 x 768)
        self.h   = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]) # 
        self.ln_f  = nn.LayerNorm(config.d_model) # Final layer norm
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size, bias=False) # language modeling head, project back to vocab size. bias=False for weight tying

        # Weight tying: share token embedding and lm_head weights
        self.lm_head.weight = self.wte.weight

        if bin_path:
            state_dict = torch.load(bin_path, map_location='cpu')
            
            # handle Transposed Weights
            transposed_keys = ['attn.c_attn.weight', 'attn.c_proj.weight', 
                               'mlp.c_fc.weight', 'mlp.c_proj.weight']
            for key in list(state_dict.keys()):
                if any(key.endswith(t_key) for t_key in transposed_keys):
                    state_dict[key] = state_dict[key].t()

            # fix the missing lm_head by copying wte.weight
            if "lm_head.weight" not in state_dict:
                state_dict["lm_head.weight"] = state_dict["wte.weight"]

            # load with strict=False to ignore the "attn.bias" buffers
            self.load_state_dict(state_dict, strict=False)
    

    def get_hidden_states(self, input_ids: Tensor) -> Tensor:
        """Returns final hidden states after ln_f, shape [B, T, d_model]."""
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(positions)
        for block in self.h:
            x, _ = block(x)
        return self.ln_f(x)

    
    def forward(
            self, 
            input_ids: Tensor, 
            past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        ) -> Tuple[CausalLMOutput, List[Tuple[Tensor, Tensor]]]: # Note the Tuple return
        """
        Forward pass of GPT-2.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            past_key_values: Optional list of past key-value pairs for KV caching

        Returns:
            CausalLMOutput with logits
        """
        # TODO: implement the GPT-2 forward pass here. 
        # The forward pass should compute the output logits for all input tokens,
        # and also update the cached attention keys and values in place (reference passing) 
        # if `past_key_values` is provided.
        B, T = input_ids.shape

        # Offset positions by the length already in the KV cache
        past_len = past_key_values[0][0].shape[2] if (past_key_values and past_key_values[0] is not None) else 0
        positions = torch.arange(past_len, past_len + T, device=input_ids.device)

        x = self.wte(input_ids) + self.wpe(positions)

        # main transformer blocks
        for i, block in enumerate(self.h):
            past_kv_i = past_key_values[i] if past_key_values else None
            x, new_kv = block(x, past_kv=past_kv_i)

            # if a cache list was provided, update it with the new KV pairs
            if past_key_values is not None:
                past_key_values[i] = new_kv

        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # return both logits
        return CausalLMOutput(logits=logits)
        
       
    def generate(
        self,
        input_ids: Tensor,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 128
    ) -> ModelOutput:
        """
        Generate tokens autoregressively using KV caching.
        
        Args:
            input_ids: [batch_size, seq_len] starting token IDs
            temperature: Sampling temperature. If 0.0, use greedy sampling.
            top_p: Top-p (nucleus) sampling threshold
            max_new_tokens: Maximum number of new tokens to generate
        
        Returns:
            ModelOutput with `sequences` containing the generated token IDs
        """        
        # TODO: implement the generation method here. 
        # You should use the `forward` method to compute logits and update KV cache at each step.
        # You can assume the input sequences are always padded to the same length,
        # and the total sequence length (input + generated) will not exceed 512 tokens.
        # GPT-2 does not have a stop token,
        # so you should always generate `max_new_tokens` new tokens 
        # for all the input sequences in the batch.
        def sample_next_token(logits: Tensor, temperature: float, top_p: float) -> Tensor:
            if temperature == 0.0:
                return logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)               # [B, vocab]

            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)

            # Mask tokens whose cumulative probability exceeds top_p,
            # keeping at least the top token (shift mask right by one)
            mask = cumulative > top_p
            mask[:, 0] = False  # always keep the top token
            sorted_probs[mask] = 0.0


            # Scatter filtered probs back to original vocabulary order
            probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            return torch.multinomial(probs, num_samples=1)  # [B, 1]
        
        # Initialize an empty list to store the KV cache for each layer
        # We populate it with None so forward() knows how many layers to expect
        past_key_values = [None] * len(self.h)


        # Prefill: process the entire prompt and build the initial KV cache
        past_key_values = [None] * len(self.h)
        output = self.forward(input_ids, past_key_values=past_key_values)
        next_token_logits = output.logits[:, -1, :]   # [B, vocab]
     
        
        # Autoregressive decode
        for _ in range(max_new_tokens):
            next_token = sample_next_token(next_token_logits, temperature, top_p) # [B, 1]
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # [B, T+1]

            # Forward on the single new token. 
            # forward() will read the existing cache and update it in-place again.
            output = self.forward(next_token, past_key_values=past_key_values)
            next_token_logits = output.logits[:, -1, :]  # [B, vocab]

        return ModelOutput(sequences=input_ids)
    


class GPT2ForSequenceClassification(nn.Module):
    """
    GPT-2 Model with a classification head.
    """

    def __init__(self, 
                 config: GPT2Config = GPT2Config(), 
                 classifier_bin_path: Optional[str] = None,
                 lm_bin_path: Optional[str] = None):
        """
        Initialize GPT-2 Classification Model.
        
        Args:
            config: GPT2Config object containing model configurations,
                    including the number of labels.
            classifier_bin_path: Path to the 
                    This file should contain the weights for 
                    both the GPT-2 base model and the classification head.
                    If empty or None,
                    the classification head weights will be initialized randomly, 
                    and the base model weights may be initialized randomly 
                    or loaded from `lm_bin_path` if provided.
            lm_bin_path: Path to the pytorch_model.bin file for the language model.
                    This file should contain the weights for the GPT-2 base model.
                    If empty or None,
                    weights may be initialized randomly, 
                    or loaded from `classifier_bin_path` if provided.
        """
        super().__init__()

        # Only one of `classifier_bin_path` and `lm_bin_path` can be provided.
        assert not (classifier_bin_path and lm_bin_path), \
            "Only one of `classifier_bin_path` and `lm_bin_path` can be provided."

        # TODO: define and initialize the GPT-2 model that can be used for sequence classification.
        # You can reuse the GPT2LMHeadModel defined above as the base model,
        # and add a classification head on top of it.
        # You should also reuse GPT2LMHeadModel's weights to speed up training if possible.

        # GPT2LMHeadModel as the backbone (optionally pre-loaded with LM weights)
        self.transformer = GPT2LMHeadModel(config, bin_path=lm_bin_path)

        # Linear classification head on top of the final hidden state
        self.classifier  = nn.Linear(config.d_model, config.num_labels, bias=False)

        # If a full classifier checkpoint is provided, load all weights
        # (backbone + head) from it, overriding anything loaded above
        if classifier_bin_path:
            state_dict = torch.load(classifier_bin_path, map_location='cpu')
            self.load_state_dict(state_dict)


    def forward(self, input_ids: Tensor) -> SequenceClassifierOutput:
        """
        Forward pass of GPT-2 for classification.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
        
        Returns:
            SequenceClassifierOutput with logits of shape (batch_size, num_labels)
        """
        
        # TODO: implement the forward pass for sequence classification here.
        # The output logits should be of shape (batch_size, num_labels),
        # where num_labels is specified in the GPT2Config,
        # and the logits contain the classification scores for each label class.
        
        # Extract hidden states from the backbone [B, T, d_model]
        hidden = self.transformer.get_hidden_states(input_ids)  # [B, T, d_model]

        sequence_lengths = (input_ids != 0).sum(dim=1) - 1 

        batch_size = input_ids.shape[0]
        cls_hidden = hidden[torch.arange(batch_size), sequence_lengths]  # [B, d_model]


        logits = self.classifier(cls_hidden)    # [B, num_labels]
        
        return SequenceClassifierOutput(logits=logits)
    
