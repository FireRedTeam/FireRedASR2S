# export PYTHONPATH=/your_path/FireRedASR2S/fireredasr2s/:$PYTHONPATH
# pip install kaldi_native_fbank
# model_dir=pretrained_models/FireRedASR2-AED

TRT_ENGINE_OUTPUT_DIR=$1


# python3 scripts/export_encoder_tensorrt.py \
#     --model-dir $model_dir \
#     --tensorrt-model-dir $TRT_ENGINE_OUTPUT_DIR \
#     --trt-engine-file-name encoder.plan || exit 1


python3 scripts/export_encoder_tensorrt.py \
    --onnx-model-path $TRT_ENGINE_OUTPUT_DIR/encoder.fp16.onnx \
    --tensorrt-model-dir $TRT_ENGINE_OUTPUT_DIR \
    --trt-engine-file-name encoder.plan


INFERENCE_PRECISION=float16
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=64
checkpoint_dir=$TRT_ENGINE_OUTPUT_DIR/tllm_checkpoint_${INFERENCE_PRECISION}
output_dir=$TRT_ENGINE_OUTPUT_DIR/trt_engine_${INFERENCE_PRECISION}

# python3 scripts/convert_checkpoint.py \
#                 --dtype ${INFERENCE_PRECISION} \
#                 --model_path $model_dir/model.pth.tar \
#                 --output_dir $checkpoint_dir || exit 1

trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 512 \
              --max_input_len 4 \
              --max_encoder_input_len 1024 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --remove_input_padding disable \
              --paged_kv_cache disable \
              --gpt_attention_plugin ${INFERENCE_PRECISION} || exit 1

# Before export_tensorrt.sh
# FireRedASR2-AED-TensorRT/
# ├── cmvn.ark
# ├── config.yaml
# ├── dict.txt
# ├── encoder.fp16.onnx
# ├── tllm_checkpoint_float16
# │   └── decoder
# │       ├── config.json
# │       └── rank0.safetensors
# └── train_bpe1000.model

# After export_tensorrt.sh
# FireRedASR2-AED-TensorRT/
# ├── cmvn.ark
# ├── config.yaml
# ├── dict.txt
# ├── encoder.fp16.onnx
# ├── encoder.plan
# ├── tllm_checkpoint_float16
# │   └── decoder
# │       ├── config.json
# │       └── rank0.safetensors
# ├── train_bpe1000.model
# └── trt_engine_float16
#     └── decoder
#         ├── config.json
#         └── rank0.engine