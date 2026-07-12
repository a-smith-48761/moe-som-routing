import sys

from transformers import GemmaTokenizerFast, Gemma3TextModel
from ..models.gemma3moe import Gemma3MoETextModel, Gemma3MoETextConfig


modelId = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-270m-it"
outputCheckpoint = sys.argv[2] if len(sys.argv) > 2 else "output/upscaled"

src = Gemma3TextModel.from_pretrained(
    modelId,
    device_map="cuda"
)

config = Gemma3MoETextConfig()
config.update (src.config.to_dict())

print ("Configuration:\n", config.to_diff_dict())

dest = Gemma3MoETextModel(config)
dest.update_from_dense (src)

print ("Saving " + outputCheckpoint)
dest.save_pretrained (outputCheckpoint)

print ("Done")

