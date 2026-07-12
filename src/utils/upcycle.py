import sys

from transformers import GemmaTokenizerFast, Gemma3TextModel
from ..models.gemma3moe import Gemma3MoETextModel, Gemma3MoETextConfig


modelId = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-270m-it"
outputCheckpoint = sys.argv[2] if len(sys.argv) > 2 else "output/upcycled"

src = Gemma3TextModel.from_pretrained(
    modelId,
    device_map="cuda"
)

config = Gemma3MoETextConfig()
config.update (src.config.to_dict())

print ("Configuration:\n", config.to_diff_dict())

# FIXME would be better if we could specify config on the command line

config.expert_geometry = [3,3]
config.expert_layer_indices = [3,4,5,6,7,8,9,10,11,12] # upcycle 10 layers in the middle of the network

dest = Gemma3MoETextModel(config)
dest.update_from_dense (src)

print ("Saving " + outputCheckpoint)
dest.save_pretrained (outputCheckpoint)

print ("Done")

