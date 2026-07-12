from dataclasses import dataclass

import torch

@dataclass
class CausalLMCollator:
    tokenizer: object

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]

        padded_labels = [
            label + [-100] * (max_length - len(label))
            for label in labels
        ]

        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long,
        )

        return batch
    
__all__ = ["CausalLMCollator"]