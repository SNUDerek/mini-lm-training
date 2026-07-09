"""
Nanotron training script example using a custom dataloader.
Based on https://github.com/huggingface/nanotron/blob/main/examples/custom-dataloader/run_train.py
Modified by Derek Hommel:
- load an already tokenized Hugging Face Dataset from disk

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=2 examples/custom-dataloader/run_train.py --config-file examples/custom-dataloader/config_custom_dl.yaml
```
"""
import argparse
from typing import Dict, cast

import datasets
from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    PretrainDatasetsArgs,
)
from nanotron.data.dataloader import get_train_dataloader
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from torch.utils.data import Dataset, DataLoader

logger = logging.get_logger(__name__)


class RepeatedDataset(Dataset):
    def __init__(self, base_dataset, total_samples: int, consumed_samples: int = 0):
        self.base_dataset = base_dataset
        self.total_samples = total_samples
        self.base_len = len(base_dataset)

        if self.base_len == 0:
            raise ValueError("base_dataset is empty")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        real_idx = idx % self.base_len
        return self.base_dataset[real_idx]




def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    num_remaining_train_steps: int,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples: The number of samples consumed by the model in the this stage (each stage starts from zero).
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert consumed_train_samples >= 0, "consumed_train_samples should be greater than 0"
    assert num_remaining_train_steps >= 0, "num_remaining_train_steps should be greater than 0"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    if isinstance(data.dataset, PretrainDatasetsArgs):
        dataset_path = data.dataset.hf_dataset_or_datasets
        if not isinstance(dataset_path, str):
            raise TypeError("hf_dataset_or_datasets must be a local dataset directory")

        log_rank(
            f"Loading pre-tokenized dataset from {dataset_path}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        base_dataset = datasets.load_from_disk(dataset_path)
        base_dataset = base_dataset.shuffle(seed=data.general.seed)
        if not isinstance(base_dataset, datasets.Dataset):
            raise TypeError(f"Expected a Dataset saved to disk, got {type(base_dataset).__name__}")

        train_dataset: Dataset = RepeatedDataset(
            base_dataset, 
            total_samples=int(trainer.general.train_steps * trainer.global_batch_size),
            consumed_samples=consumed_train_samples,
        )

        dataloader = get_train_dataloader(
            train_dataset=train_dataset,
            sequence_length=trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            seed_worker=data.seed,
            dataloader_drop_last=True,
        )

        total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
        num_tokens_needed_for_training = (
            num_remaining_train_steps * trainer.global_batch_size * trainer.sequence_length
        )
        assert num_tokens_needed_for_training <= total_tokens_dataset, (
            f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
            f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.iteration_step}"
        )
    else:
        raise ValueError(f"Expected PretrainDatasetsArgs, got {data.dataset}")

    return dataloader


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples, _ = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, trainer.metadata)
        assert (
            consumed_train_samples is not None
        ), f"Cannot find consumed_train_samples for stage {stage.start_training_step} in the checkpoint"

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"[Training Plan] Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
            if stage_idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
