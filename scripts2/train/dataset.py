"""Token dataset for training language models.

Token Dataset uses WebDataset to stream tokenized audio data from .tar files.
It yields fixed-size blocks of tokens for training. Token sequences are packed.

"""

import os
import random
import traceback
from pathlib import Path
import webdataset as wds

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset


# CONSTANTS
SEED = 42
MAX_TOKENS = 2048
# PAD_TOKEN_ID = 2000
BOS_TOKEN_ID = 2000
EOS_TOKEN_ID = 2001
VOCAB_SIZE = 2002  # 2000 tokens + BOS + EOS
SHUFFLE_BUFFER = 1000


class TokenDataset(IterableDataset):
    def __init__(self, tokens_dir: str):

        # self.urls is all the paths strings to *.tar files in tokens_dir
        self.urls = sorted([str(p) for p in Path(tokens_dir).glob("*.tar")])

        self.block_size = MAX_TOKENS
        self.shuffle_buffer = SHUFFLE_BUFFER

        random.seed(SEED)
        random.shuffle(self.urls)

    def __iter__(self):
        dataset = (
            wds.WebDataset(  # type: ignore
                self.urls,
                resampled=True,
                shardshuffle=True,
                nodesplitter=wds.shardlists.split_by_node,
            )
            # .split_by_node()
            # .split_by_worker()
            .shuffle(self.shuffle_buffer).decode()
        )
        dataset = dataset.compose(wds.shardlists.split_by_worker)

        buffer = []

        for sample in dataset:
            tokens = sample.get("tokens.npy")

            if tokens is None:
                print("Warning: 'tokens.npy' not found in sample, skipping.")
                continue  # Skip samples without tokens

            # token_tensor = torch.from_numpy(tokens).long()

            token_list = [BOS_TOKEN_ID] + tokens.tolist() + [EOS_TOKEN_ID]
            buffer.extend(token_list)

            # when we have enough tokens, cut and yield a block
            while len(buffer) >= self.block_size:
                block = buffer[: self.block_size]
                buffer = buffer[self.block_size :]

                yield {"input_ids": torch.tensor(block, dtype=torch.long)}

            # if PAD_TOKEN_ID in token_tensor:
            #     print(f"Found {PAD_TOKEN_ID}: {token_tensor}")
            #     raise ValueError("PAD_TOKEN_ID found in token tensor!")

            # if len(token_tensor) > self.max_tokens:  # Count # of occurences of this
            #     continue  # Skip sequences that are too long for GPU memory

            # yield {"input_ids": token_tensor, "length": len(token_tensor)}


def collate_fn(batch):
    tensors = torch.stack([item["input_ids"] for item in batch])  # [B, MAX_TOKENS]

    input_ids = tensors[:, :-1].contiguous()
    labels = tensors[:, 1:].contiguous()
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

    # tensors = [sample["input_ids"] for sample in batch]

    # # Pad sequences
    # input_ids = pad_sequence(tensors, batch_first=True, padding_value=PAD_TOKEN_ID)
    # labels = input_ids.clone().masked_fill(input_ids == PAD_TOKEN_ID, -100)

    # # Shift input_ids and labels
    # input_ids = input_ids[:, :-1].contiguous()
    # labels = labels[:, 1:].contiguous()

    # # Adjust lengths to account for shift
    # lengths = torch.tensor([sample["length"] for sample in batch]) - 1

    # # Create attention mask
    # max_length = input_ids.size(1)
    # indices = torch.arange(max_length)
    # # print(f"Indices shape: {indices.shape}, Lengths shape: {lengths.shape}")
    # attention_mask = (indices < lengths.unsqueeze(1)).int()

    # # pad_indices = torch.arange(max_length).expand(len(batch), max_length)
    # # attention_mask = (pad_indices < lengths.unsqueeze(1)).int()
    # # print(f"Attention mask shape: {attention_mask.shape}")

    # return {
    #     "input_ids": input_ids,
    #     "labels": labels,
    #     "lengths": lengths,
    #     "attention_mask": attention_mask,
    # }


if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, running on CPU")

    print(f"[Rank {rank}] Initialized (Local Rank: {local_rank}, World: {world_size})")

    BATCH_SIZE = 4  # Small batch size for testing
    tokens_dir = "/scratch2/ddager/amela/tokens/chunks5_mhubert/"

    try:
        dataset = TokenDataset(tokens_dir)

        # We use a simple DataLoader.
        # Note: In DDP, we do NOT need a DistributedSampler because WebDataset
        # handles splitting via .split_by_node() in __iter__
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=2,  # Testing with workers to ensure multiprocessing works
            pin_memory=True,
        )

        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            lengths = batch["lengths"]
            mask = batch["attention_mask"]

            # Stop after 1 batch for this test
            break

    except Exception as e:
        print(f"[Rank {rank}] CRITICAL ERROR: {e}")
        traceback.print_exc()

    finally:
        # Cleanup DDP
        if dist.is_initialized():
            dist.destroy_process_group()

    # inspect_shards(tokens_dir)

    # dataset = TokenDataset(tokens_dir)
    # it = iter(dataset)
    # first_batch = next(it)
    # print(f"\nDataset yielded first item with shape: {first_batch['input_ids'].shape}")


# def inspect_shards(tokens_dir):
#     """Helper to see exactly what keys and data types are in your shards."""
#     urls = [str(p) for p in Path(tokens_dir).glob("*.tar")]

#     rank = int(os.environ.get("RANK", 0))

#     if rank == 0:
#         print(f"\n[Rank {rank}] Inspecting first shard: {urls[0]}")
#         dataset = wds.WebDataset(  # type: ignore
#             urls[0], nodesplitter=None, shardshuffle=False
#         ).decode()

#         # Take the first 2 samples to inspect
#         for i, sample in enumerate(dataset):
#             if i >= 5:
#                 break

#             print(f"\nSample {i}:")
#             print(f"  Internal Basename (key): {sample.get('__key__')}")
#             print(f"  Available Keys in Dict: {list(sample.keys())}")

#             for key, value in sample.items():
#                 if key == "__key__":
#                     continue
#                 v_type = type(value)
#                 shape = value.shape if hasattr(value, "shape") else "N/A"
#                 print(f"    -> Key: '{key}' | Type: {v_type} | Shape/Len: {shape}")

#                 if key.endswith(".json"):
#                     print(f"Contents of {key}:")
#                     print(value)  # This prints the actual dict from the json file
