import sys

from transformers import GemmaTokenizerFast, Gemma3TextModel
from ..models.gemma3moe import Gemma3MoETextModel, Gemma3MoETextConfig


modelId = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-270m-it"
outputCheckpoint = sys.argv[2] if len(sys.argv) > 2 else "output/upcycled"

src = Gemma3TextModel.from_pretrained(
    modelId,
    device_map="cuda"
)
src_tokenizer = GemmaTokenizerFast.from_pretrained (modelId)

src_config_dict = src.config.to_dict()
src_config_dict.pop ("model_type", None)
src_config_dict.pop ("architectures", None)

config = Gemma3MoETextConfig(**src_config_dict, 
                            expert_geometry = [3,3],                   # 3x3 array of experts
                            expert_layer_indices = list(range(2,16)),  # upcycle 14 layers in the middle of the network
                            expert_router_training_type="gradient")    # default to gradient training

print ("Configuration:\n", config.to_diff_dict())


dest = Gemma3MoETextModel(config)
dest.update_from_dense (src)

print ("Saving " + outputCheckpoint)
dest.save_pretrained (outputCheckpoint)
src_tokenizer.save_pretrained (outputCheckpoint)

print ("Done")

