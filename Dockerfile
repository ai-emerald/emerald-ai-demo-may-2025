# Copyright (c) 2025 Emerald AI
# SPDX-License-Identifier: Apache-2.0
FROM mosaicml/llm-foundry:2.5.1_cu124-d91c4be@sha256:4fb4ae74869918a2336298ecfa5ae1623ecd1d5a717b795c5cdbf690cdcc3641

RUN git clone https://github.com/mosaicml/llm-foundry.git && \
    cd llm-foundry && \
    git checkout d91c4beb0161174b5e915b5aaa49c09003c3599d

WORKDIR /llm-foundry
ADD llm-foundry-d91c4beb-add-emerald.patch .
RUN git apply llm-foundry-d91c4beb-add-emerald.patch

# Requires NVIDIA GPU
RUN pip install -e ".[gpu]"  

CMD ["/bin/bash"]
