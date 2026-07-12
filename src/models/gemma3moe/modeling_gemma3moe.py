
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_func_from_hub, use_kernelized_func
from transformers.masking_utils import (
    _preprocess_mask_arguments,
    blockwise_overlay,
    create_causal_mask,
    create_masks_for_generate,
    create_sliding_window_causal_mask,
    maybe_pad_block_sequence_ids,
    sliding_window_overlay,
)
from transformers.modeling_layers import GenericForSequenceClassification, GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.models.auto import AutoModel
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3TextScaledWordEmbedding,
    Gemma3MLP,
    Gemma3DecoderLayer,
    Gemma3RotaryEmbedding,
    Gemma3Attention,
    Gemma3PreTrainedModel,
    Gemma3RMSNorm,
    Gemma3TextModel
)
from .configuration_gemma3moe import Gemma3MoETextConfig


class Gemma3MoEMLP(nn.Module):
    """
    A mixture of experts MLP layer with configurable router update rule.
    Each expert is a standard MLP layer, and the router is a linear layer that outputs a distribution over the experts.
    Router update may depend on a 2-dimensional array of experts, with the location of expert (i,j) being stored at index i * num_experts_per_row + j.
     The router update rule is specified by the `expert_router_training_type` parameter in the config, which can be either "som" or "gradient".
    """
    def __init__(self, config: Gemma3MoETextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.expert_geometry[0] * config.expert_geometry[1]
        self.experts = nn.ModuleList([Gemma3MLP(config) for x in range(self.num_experts)])
        self.gate_proj = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.topk = config.expert_router_topk
        self.update_rule = config.expert_router_training_type

    def update_from_dense(
        self,
        dense_mlp: Gemma3MLP,
    ) -> "Gemma3MoEMLP":
        """
        Convert a traditional Gemma3MLP to an MoE equivalent by duplicating its weights
        """

        for expert in self.experts:
            expert.load_state_dict(dense_mlp.state_dict())

    
    def forward(self, x):
        # x's shape is (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = x.shape

        # Compute the router logits (shape: (batch_size, seq_len, num_experts))
        router_logits = self.gate_proj(x)
        # Compute the indices and values for the top-k experts for each token (shape: (batch_size, seq_len, topk))
        topk_values, topk_indices = torch.topk(router_logits, self.topk, dim=-1)
        # Compute the weights for each expert using softmax over the top-k experts (shape: (batch_size, seq_len, topk))
        topk_softmax = torch.softmax(topk_values, dim=-1)

        # Create a weight matrix to combine the outputs of the top-k experts (shape: (batch_size, seq_len, num_experts))
        weight_matrix = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device, dtype=topk_softmax.dtype)
        weight_matrix.scatter_(dim=2, index=topk_indices, src=topk_softmax)

        # Initialize the output tensor
        output = torch.zeros_like(x)

        # Combine the outputs of all experts weighted by the weight matrix
        for i in range(self.num_experts):
            # Find indices of tokens that are routed to expert i
            indices = torch.where(weight_matrix[..., i] > 0) # tuple of (batch_indices, seq_indices)
            if len(indices[0]) != 0:
                # Route the tokens to the expert and compute the output
                expert_output = self.experts[i](x[indices]) # shape: (num_tokens_routed_to_expert_i, hidden_size)
                
                # Weight the expert output by the corresponding weights
                weighted_expert_output = expert_output * weight_matrix[indices[0], indices[1], i].unsqueeze(-1) # shape: (num_tokens_routed_to_expert_i, hidden_size)

                # Scatter the weighted expert output back to the output tensor
                output[indices] += weighted_expert_output
                
        return output



class Gemma3MoEDecoderLayer(GradientCheckpointingLayer):
    """
    A clone of the Gemma3DecoderLayer, but with a mixture of experts MLP layer instead of a standard MLP layer.
    The mixture of experts MLP layer is only used if the layer index is in the list of expert layer indices specified in the config. Otherwise, a standard MLP layer is used.
    """
    def __init__(self, config: Gemma3MoETextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        if layer_idx in config.expert_layer_indices:
            self.mlp = Gemma3MoEMLP(config)
        else:
            self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def update_from_dense(
        self,
        dense_decoder: Gemma3DecoderLayer
    ) -> "Gemma3MoEMLP":
        """
        Convert a traditional Gemma3DecoderLayer to an MoE equivalent by copying its attention layer
        """

        # copy state from dense mlp to the new module's mlp/experts
        dense_mlp = dense_decoder.mlp
        if isinstance(self.mlp, Gemma3MoEMLP):
            self.mlp.update_from_dense(dense_mlp)
        else:
            self.mlp.load_state_dict (dense_mlp.state_dict())

        # copy state from attention to the new module
        self.self_attn.load_state_dict (dense_decoder.self_attn.state_dict())
        # and copy normalisation state
        self.input_layernorm.load_state_dict (dense_decoder.input_layernorm.state_dict())
        self.post_attention_layernorm.load_state_dict (dense_decoder.post_attention_layernorm.state_dict())
        self.pre_feedforward_layernorm.load_state_dict (dense_decoder.pre_feedforward_layernorm.state_dict())
        self.post_feedforward_layernorm.load_state_dict (dense_decoder.post_feedforward_layernorm.state_dict())
                                                          
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states



def _bidirectional_window_overlay(sliding_window: int) -> Callable[[int, int, int, int], bool]:
    """
    Enables a bidirectional mask within the sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """A token can attend to any other token if their absolute distance is within
        the (exclusive) sliding window size (distance < sliding_window)."""
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


@auto_docstring
class Gemma3MoETextModel(Gemma3PreTrainedModel):
    config: Gemma3MoETextConfig
    input_modalities = ("text",)

    def __init__(self, config: Gemma3MoETextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3MoE downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3MoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def update_from_dense (self, model : Gemma3TextModel):
        """
        Copy weights from an existing dense model into all appropriate weights in this model, including replacing dense layers into each expert for layers using a mixture of experts.
        """
        if len(model.layers) != len(self.layers):
            raise ValueError(f"Model layer count does not match [${len(model.layers)} rather than ${len(self.layers)}]")
        
        # convert layers where necessary:
        for i in range(len(model.layers)):
            self.layers[i].update_from_dense (model.layers[i])
        # these don't need conversion so can just be copied:
        self.norm.load_state_dict(model.norm.state_dict())
        self.embed_tokens.load_state_dict(model.embed_tokens.state_dict())
        
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            if self.config.use_bidirectional_attention:
                mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(self.config.sliding_window)

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class Gemma3MoEForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3MoETextConfig

    def __init__(self, config: Gemma3MoETextConfig):
        super().__init__(config)
        self.model = Gemma3MoETextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3MoEForCausalLM

        >>> model = Gemma3MoEForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




__all__ = [
    "Gemma3MoETextModel",
    "Gemma3MoEForCausalLM",
]
