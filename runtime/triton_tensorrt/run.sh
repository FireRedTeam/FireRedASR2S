stage=$1
stop_stage=$2
export CUDA_VISIBLE_DEVICES="0"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading FireRedASR2-AED-TensorRT"
    huggingface-cli download yuekai/FireRedASR2-AED-TensorRT --local-dir ./FireRedASR2-AED-TensorRT
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "export TensorRT engines"
    bash scripts/export_tensorrt.sh ./FireRedASR2-AED-TensorRT
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Starting Triton server"
    tritonserver --model-repository=model_repo_fireredasr2_aed
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "http client"
    wget -nc https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_zh/wav/long.wav
    python3 http_client.py --wav_path long.wav
fi


if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Running Triton client"
    if [ ! -d "Triton-ASR-Client" ]; then
        git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
    fi
    num_task=128
    dataset_name=yuekai/aishell
    subset_name=test
    split_name=test


    dataset_name="yuekai/speechio"
    subset_name="SPEECHIO_ASR_ZH00007"
    split_name="test"

    python3 Triton-ASR-Client/client.py \
        --server-addr localhost \
        --model-name fireredasr \
        --num-tasks $num_task \
        --log-dir ./log_fireredasr_$subset_name_$num_task \
        --huggingface_dataset $dataset_name \
        --subset_name $subset_name \
        --split_name $split_name \
        --compute-cer
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Running offline inference"
    export DISABLE_TORCH_DEVICE_SET=True
    dataset_name=yuekai/aishell
    subset_name=test
    split_name=test

    # dataset_name="yuekai/speechio"
    # subset_name="SPEECHIO_ASR_ZH00007"
    # split_name="test"

    torchrun --nproc_per_node=1 \
        infer.py \
        --engine_dir ./FireRedASR2-AED-TensorRT \
        --huggingface_dataset $dataset_name \
        --subset_name $subset_name \
        --split_name $split_name \
        --ctc ./FireRedASR2-AED-TensorRT/ctc.pt \
        --batch_size 64

fi