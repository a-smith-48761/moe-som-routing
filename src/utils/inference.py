import sys

from transformers import GemmaTokenizerFast
from ..models.gemma3moe import Gemma3MoEForCausalLM


modelId = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-270m-it"

model = Gemma3MoEForCausalLM.from_pretrained(
    modelId,
    device_map="cuda",
    attn_implementation="sdpa" # scalde dot product attention, autodetect best available implementation
)

tokenizer = GemmaTokenizerFast.from_pretrained (modelId)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Tell me why the sky is blue"},
        ]
    },
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=200, cache_implementation="dynamic")
print(tokenizer.decode(output[0], skip_special_tokens=True))

