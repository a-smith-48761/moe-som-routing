import sys

from transformers import GemmaTokenizerFast
from ..models.gemma3moe import Gemma3MoEForCausalLM


modelId = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-270m-it"

model = Gemma3MoEForCausalLM.from_pretrained(
    modelId,
    device_map="cuda",
    attn_implementation="sdpa" # scalde dot product attention, autodetect best available implementation
)

for name, module in model.named_modules():
   print (name, type(module), module.extra_repr())

