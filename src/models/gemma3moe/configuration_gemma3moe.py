from typing import (Any, List)

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import auto_docstring, logging
from transformers.models.siglip import SiglipVisionConfig


logger = logging.get_logger(__name__)


@strict
class Gemma3MoETextConfig(PreTrainedConfig):
    r"""
    query_pre_attn_scalar (`float`, *optional*, defaults to 256):
        scaling factor used on the attention scores
    final_logit_softcapping (`float`, *optional*):
        Scaling factor when applying tanh softcapping on the logits.
    attn_logit_softcapping (`float`, *optional*):
        Scaling factor when applying tanh softcapping on the attention scores.
    use_bidirectional_attention (`bool`, *optional*, defaults to `False`):
        If True, the model will attend to all text tokens instead of using a causal mask. This does not change
        behavior for vision tokens.
    expert_geometry (`List[int]`, *optional*, defaults to `[3,3]`):
        Horizontal and vertical dimensions of the grid of experts used for each expert layer. If using the
        gradient update rule for routing the precise shape does not matter other than to determine the total
        number of experts, but for SOM updates the shape is used during training.
    expert_layer_indices (`List[int]`, *optional*):
        List of layer indices to use mixture of experts for. Any layer not in this list will be initialised only
        with a single MLP rather than the number of experts listed above. By default, no layers are chosen.
    expert_router_training_type (`str`, *optional*, defaults to `'gradient'`):
        Either `'gradient'` to use traditional gradient-based training of the routing network or `'som'` to use
        training based on Self-Organizing Maps (Kohonen, *The Self-Organizing Map*, 1990).
    expert_router_topk (`int`, *optional*, defaults to `2`):
        The number of experts to activate for each input to each layer.
    """

    model_type = "gemma3moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 262_208
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    query_pre_attn_scalar: int = 256
    sliding_window: int | None = 4096
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None
    use_bidirectional_attention: bool | None = False

    # MoE-specific parameters
    expert_geometry: List[int] | None = None     # geometry of each MoE layer, which defaults to [3,3] for a 3x3 layer if set to None.
    expert_layer_indices: List[int] | None = None  # indices of the layers that are MoE layers, which defaults to no layers being MoE layers if set to None.
    expert_router_training_type: str = "som"     # either "som" or "gradient"
    expert_router_topk: int = 2 

    default_theta = {"global": 1_000_000.0, "local": 10_000.0}

    def __post_init__(self, **kwargs):
        if self.use_bidirectional_attention:
            self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds

        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.expert_geometry is None:
            self.expert_geometry = [3, 3]  # default to a 3x3 MoE layer
        if self.expert_layer_indices is None:
            self.expert_layer_indices = []  # default to no MoE layers

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`. If we find `rope_parameters`
        # as arg in the inputs, we can safely assume that it is in the new format. New naming used -> new format
        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)

        # Set default values if not present
        if self.rope_parameters.get("full_attention") is None:
            self.rope_parameters["full_attention"] = {"rope_type": "default"}
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_theta", self.default_theta["global"])
        )
        if self.rope_parameters.get("sliding_attention") is None:
            self.rope_parameters["sliding_attention"] = {"rope_type": "default"}
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_local_base_freq", self.default_theta["local"])
        )

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.standardize_rope_params()
        return kwargs


__all__ = ["Gemma3MoETextConfig"]
