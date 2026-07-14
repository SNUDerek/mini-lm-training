import torch
import torch.nn.functional as F

from transformers import PreTrainedModel, Trainer


class PackedCausalTrainer(Trainer):
    """
    Trainer for already-shifted next-token labels.

    LlamaForCausalLM normally shifts labels internally. Because our collator
    already constructs labels[t] = input_ids[t + 1], we calculate the loss
    directly from unshifted model logits.
    """

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        if logits.shape[:2] != labels.shape:
            raise RuntimeError(
                f"Logits shape {tuple(logits.shape)} is incompatible with "
                f"labels shape {tuple(labels.shape)}"
            )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )

        return (loss, outputs) if return_outputs else loss