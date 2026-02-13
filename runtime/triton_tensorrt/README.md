# Accelerating FireRedASR2-AED with TensorRT-LLM

This guide covers two inference modes:
1. **Offline Inference** - Direct local inference without server setup
2. **Client-Server Mode** - Scalable deployment with NVIDIA Triton Inference Server

---

## Offline Inference


### Setup

```sh
uv sync
```

### Export TensorRT Engine

```sh
huggingface-cli download yuekai/FireRedASR2-AED-TensorRT --local-dir ./FireRedASR2-AED-TensorRT
uv run bash scripts/export_tensorrt.sh ./FireRedASR2-AED-TensorRT
```

### Run Inference

```sh
uv run torchrun infer.py --huggingface_dataset yuekai/aishell
```

### Benchmark Results

Decoding AISHELL-1 test set (10 hours) on a single H20 GPU:

| Method | Time (secs) |
| :--- | :--- |
| PyTorch Offline Inference | 760 |
| TensorRT Offline Inference | 60 |

**~12.7x speedup** with TensorRT acceleration.

---

## Client-Server Mode

Deploy FireRedASR-AED as a scalable service using NVIDIA Triton Inference Server.

### Quick Start

Launch the service directly with Docker Compose:
```sh
HF_TOKEN="hf_your_token" docker compose up
```

### Build the Docker Image
To use the pre-built docker image:
```sh
docker pull soar97/triton-fireredasr:25.06
```

To build the image from scratch:
```sh
docker build . -f Dockerfile -t soar97/triton-fireredasr:25.06
```

### Run a Docker Container

```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "fireredasr-aed-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-fireredasr:25.06
```

### Understanding `run.sh`

The `run.sh` script orchestrates the entire workflow through numbered stages.

Run a subset of stages with:
```sh
bash run.sh <start_stage> <stop_stage>
```

**Stages:**

| Stage | Description |
| :--- | :--- |
| 0 | Download Models |
| 1 | Export TensorRT Engines |
| 2 | Start Triton Server |
| 3 | HTTP Client Test |
| 4 | Benchmark with Triton Client |
| 5 | Offline Inference Benchmark |

### Export Models and Launch Server

Inside the Docker container, prepare the models and start the Triton server:
```sh
# Run stages 0, 1, and 2
bash run.sh 0 2
```

### Single-Utterance HTTP Client

Send a single HTTP inference request:
```sh
bash run.sh 3 3
# Or directly:
python3 http_client.py --wav_path long.wav
```

### Benchmark with Triton Client

Benchmark the running Triton server:
```sh
bash run.sh 4 4
```
This will clone `Triton-ASR-Client` and run evaluation on the `yuekai/aishell` dataset.
