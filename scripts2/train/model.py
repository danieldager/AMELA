import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from scripts2.train.dataset import (
    TokenDataset,
    collate_fn,
    MAX_TOKENS,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    VOCAB_SIZE,
)


# --- CUSTOM CALLBACK FOR LOGGING ---
class DebugCallback(TrainerCallback):
    """Prints a clear message at the start of training to confirm things are working."""

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 1 and state.is_world_process_zero:
            print("\n" + "=" * 40)
            print("  TRAINING STARTED SUCCESSFULLY  ")
            print("  (First batch processed)        ")
            print("=" * 40 + "\n")


if __name__ == "__main__":
    tokens_dir = "/scratch2/ddager/amela/tokens/chunks5_mhubert/"
    dataset = TokenDataset(tokens_dir)

    config = GPT2Config(
        vocab_size=VOCAB_SIZE,  # 2002 tokens (2000 + BOS + EOS)
        n_positions=MAX_TOKENS,  # 2048 (~40 seconds of audio at 50Hz)
        n_ctx=MAX_TOKENS,  # Context window
        n_embd=768,
        n_layer=12,
        n_head=12,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        # pad_token_id=PAD_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,  # type: ignore
        eos_token_id=EOS_TOKEN_ID,  # type: ignore
    )

    model = GPT2LMHeadModel(config)

    training_args = TrainingArguments(
        output_dir="./checkpoints/gpt2_test",
        overwrite_output_dir=True,
        # Optimization
        per_device_train_batch_size=32,  # Adjusted from your snippet
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_grad_norm=1.0,  # Good default
        weight_decay=0.01,
        # Scheduling
        lr_scheduler_type="cosine",  # Cosine is usually better than inverse_sqrt for Transformers
        warmup_steps=100,
        max_steps=50,  # LOW FOR TESTING (Increase later)
        # Precision & Speed
        bf16=True,  # Enable BF16
        dataloader_num_workers=4,
        # Logging & Saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        # DDP
        ddp_find_unused_parameters=False,  # Optimization for standard models
        remove_unused_columns=False,  # Important since we use 'input_ids' which isn't in signature? No, 'input_ids' IS in signature.
        # Actually we set this to False so 'lengths' doesn't get stripped before collate,
        # but usually Trainer strips AFTER collate.
        # We will trust ignore_keys list below.
        # Misc
        label_names=["labels"],  # Ensures Trainer knows which key is the target
        ignore_data_skip=True,  # Faster restart for IterableDatasets
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[DebugCallback()],
    )

    trainer.train()
    print("Training complete.")
