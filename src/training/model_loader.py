from ..models.gemma3moe import Gemma3MoEForCausalLM
from transformers import Gemma3ForCausalLM

def load_model(model_id: str):
    if model_id.startswith("google/"):
        return Gemma3ForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            attn_implementation="sdpa",
        )

    return Gemma3MoEForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        attn_implementation="sdpa",
    )

