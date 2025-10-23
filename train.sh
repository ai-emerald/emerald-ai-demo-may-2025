#!/usr/bin/env bash

# Copyright (c) 2025 Emerald AI
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

docker build -t llm-foundry .
docker volume create datasets
docker run -it --rm --gpus all --privileged --shm-size 2G \
    -v datasets:/datasets \
    -e MEMCACHED_HOST -e EMERALD_CONTROL_NAME \
    llm-foundry bash -c "
        python emerald/control_power.py &
        
        # Minimal llm-foundry training example, fits on a T4
        cd scripts
        if [ ! -d /datasets/my-copy-c4 ]; then
            python data_prep/convert_dataset_hf.py \
                --dataset allenai/c4 --data_subset en \
                --out_root /datasets/my-copy-c4 --splits train_small val_xsmall \
                --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'
        fi
        composer train/train.py \
            train/yamls/pretrain/mpt-125m.yaml \
            variables.data_local=/datasets/my-copy-c4 \
            train_loader.dataset.split=train_small \
            eval_loader.dataset.split=val_xsmall \
            eval_interval=0 \
            save_folder=mpt-125m \
            global_train_batch_size=64 \
            device_train_microbatch_size=1 \
            model.attn_config.attn_impl=torch \
            precision=amp_fp16
    "
