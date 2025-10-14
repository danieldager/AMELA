import argparse
import os
from multiprocessing import Process

import fairseq.data.dictionary  # type: ignore
import torch
import torchaudio  # type: ignore
from textless.data.speech_encoder import SpeechEncoder  # type: ignore
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder  # type: ignore
from textless.vocoders.tacotron2.vocoder import TacotronVocoder  # type: ignore

# Allowlist fairseq classes for PyTorch 2.6+ security
torch.serialization.add_safe_globals(
    [
        argparse.Namespace,
        fairseq.data.dictionary.Dictionary,
    ]
)

# All configurations
ALL_CONFIGS = [
    ("tacotron", "hubert-base-ls960", "kmeans", 50),
    ("tacotron", "hubert-base-ls960", "kmeans", 100),
    ("tacotron", "hubert-base-ls960", "kmeans", 200),
    ("tacotron", "cpc-big-ll6k", "kmeans", 50),
    ("tacotron", "cpc-big-ll6k", "kmeans", 100),
    ("tacotron", "cpc-big-ll6k", "kmeans", 200),
    ("hifigan", "mhubert-base-25hz", "kmeans", 500),
    ("hifigan", "hubert-base-ls960-layer-9", "kmeans", 500),
    ("hifigan", "hubert-base-ls960-layer-9", "kmeans-expresso", 2000),
    ("hifigan", "mhubert-base-vp_mls_cv_8lang", "kmeans", 2000),
    ("hifigan", "mhubert-base-vp_mls_cv_8lang", "kmeans-expresso", 2000),
]

FILES = [
    "/store/projects/lexical-benchmark/audio/symlinks/50h/05/1087_LibriVox_en_seq_058.wav",  # 120s
    "/store/projects/lexical-benchmark/audio/symlinks/50h/05/1087_LibriVox_en_seq_010.wav",  # 30s
    "/store/projects/lexical-benchmark/audio/symlinks/50h/05/1087_LibriVox_en_seq_000.wav",  # 10s
    "/store/projects/lexical-benchmark/audio/symlinks/50h/05/1087_LibriVox_en_seq_002.wav",  # 5s
]


def process_config(gpu_id, configs):
    """Process a subset of configs on a specific GPU"""
    # Set which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for vocoder_name, dense_model, quantizer, vocab_size in configs:
        encoder = SpeechEncoder.by_name(
            dense_model_name=dense_model,
            quantizer_model_name=quantizer,
            vocab_size=vocab_size,
            deduplicate=True,
        ).cuda()

        if vocoder_name == "tacotron":
            vocoder = TacotronVocoder.by_name(
                dense_model,
                quantizer,
                vocab_size,
            ).cuda()
        elif vocoder_name == "hifigan":
            vocoder = CodeHiFiGANVocoder.by_name(
                dense_model,
                quantizer,
                vocab_size,
            ).cuda()

        vocod = vocoder_name[:4]
        model = dense_model.split("-")[0]
        config = f"{vocod}_{model}_{quantizer}_{vocab_size}"
        print(f"[GPU {gpu_id}] Testing: {config}")

        for file in FILES:
            waveform, sr = torchaudio.load(file)
            encoded = encoder(waveform.cuda())
            units = encoded["units"]

            audio = vocoder(units)
            torchaudio.save(
                f"output/{config}_{file.split('/')[-1]}",
                audio.cpu().float().unsqueeze(0),
                vocoder.output_sample_rate,
            )

        print(f"[GPU {gpu_id}] Completed: {config}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
    args = parser.parse_args()

    num_gpus = args.num_gpus
    configs_per_gpu = len(ALL_CONFIGS) // num_gpus

    processes = []
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * configs_per_gpu
        end_idx = (
            start_idx + configs_per_gpu if gpu_id < num_gpus - 1 else len(ALL_CONFIGS)
        )
        gpu_configs = ALL_CONFIGS[start_idx:end_idx]

        p = Process(target=process_config, args=(gpu_id, gpu_configs))
        p.start()
        processes.append(p)
        print(f"Started GPU {gpu_id} with {len(gpu_configs)} configs")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All processing complete!")


if __name__ == "__main__":
    main()
