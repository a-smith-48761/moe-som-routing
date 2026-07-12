import sys

from transformers import GemmaTokenizerFast, Gemma3ForCausalLM, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

from ..models.gemma3moe import Gemma3MoEForCausalLM


modelId = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-270m-it"
datasetId = sys.argv[2] if len(sys.argv) > 2 else "openai/gsm8k"
datasetSplit = sys.argv[3] if len(sys.argv) > 3 else "test"
datasetConfig = sys.argv[4] if len(sys.argv) > 4 else "main"

if modelId.startswith("google/"):
    # official model, so use the standard class
    model = Gemma3ForCausalLM.from_pretrained(
        modelId,
        device_map="cuda",
        attn_implementation="sdpa" # scalde dot product attention, autodetect best available implementation
    )
else:    
    model = Gemma3MoEForCausalLM.from_pretrained(
        modelId,
        device_map="cuda",
        attn_implementation="sdpa" # scalde dot product attention, autodetect best available implementation
    )

tokenizer = GemmaTokenizerFast.from_pretrained (modelId)

trainer = Trainer  (
    model = model,
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer),
    processing_class = tokenizer
)
metrics = trainer.evaluate (eval_dataset = load_dataset(datasetId, datasetConfig, split=datasetSplit))

print (metrics)
