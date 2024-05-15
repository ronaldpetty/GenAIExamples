#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_name=$(echo $(hostname) | tr '[a-z]-' '[A-Z]_')_$(echo 'IP')
ip_address=$(eval echo '$'$ip_name)

function build_docker_images() {
    cd $WORKPATH
    git clone https://github.com/opea-project/GenAIComps.git
    cd GenAIComps

    docker build -t opea/gen-ai-comps:embedding-tei-server -f comps/embeddings/langchain/docker/Dockerfile .
    docker build -t opea/gen-ai-comps:retriever-redis-server -f comps/retrievers/langchain/docker/Dockerfile .
    docker build -t opea/gen-ai-comps:reranking-tei-server -f comps/reranks/langchain/docker/Dockerfile .
    docker build -t opea/gen-ai-comps:llm-tgi-gaudi-server -f comps/llms/langchain/docker/Dockerfile .

    cd ..
    git clone https://github.com/huggingface/tei-gaudi
    cd tei-gaudi/
    docker build --no-cache -f Dockerfile-hpu -t opea/tei-gaudi .

    docker pull ghcr.io/huggingface/tgi-gaudi:1.2.1
    docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-1.2

    cd $WORKPATH/microservice/gaudi
    docker build --no-cache -t opea/gen-ai-comps:chatqna-megaservice-server -f docker/Dockerfile .

    cd $WORKPATH/ui
    docker build --no-cache -t opea/gen-ai-comps:chatqna-ui-server -f docker/Dockerfile .

    docker images
}

function start_services() {
    cd $WORKPATH/microservice/gaudi

    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export RERANK_MODEL_ID="BAAI/bge-reranker-large"
    export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:8090"
    export TEI_RERANKING_ENDPOINT="http://${ip_address}:8808"
    export TGI_LLM_ENDPOINT="http://${ip_address}:8008"
    export REDIS_URL="redis://${ip_address}:6379"
    export INDEX_NAME="rag-redis"
    export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
    export MEGA_SERVICE_HOST_IP=${ip_address}
    export BACKEND_SERVICE_ENDPOINT="http://${ip_address}:8888/v1/chatqna"

    # Start Docker Containers
    # TODO: Replace the container name with a test-specific name
    docker compose -f docker_compose.yaml up -d

    sleep 1m # Waits 1 minutes
}

function validate_microservices() {
    # Check if the microservices are running correctly.
    # TODO: Any results check required??
    curl ${ip_address}:8090/embed \
        -X POST \
        -d '{"inputs":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json' > ${LOG_PATH}/embed.log
    sleep 5s

    curl http://${ip_address}:6000/v1/embeddings \
        -X POST \
        -d '{"text":"hello"}' \
        -H 'Content-Type: application/json' > ${LOG_PATH}/embeddings.log
    sleep 10s

    export PATH="${HOME}/miniconda3/bin:$PATH"
    source activate
    test_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
    curl http://${ip_address}:7000/v1/retrieval \
        -X POST \
        -d '{"text":"test","embedding":${your_embedding}}' \
        -H 'Content-Type: application/json' > ${LOG_PATH}/retrieval.log
    sleep 5s

    curl http://${ip_address}:8808/rerank \
        -X POST \
        -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
        -H 'Content-Type: application/json' > ${LOG_PATH}/rerank.log
    sleep 5s

    curl http://${ip_address}:8000/v1/reranking \
        -X POST \
        -d '{"initial_query":"What is Deep Learning?", "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}]}' \
        -H 'Content-Type: application/json' > ${LOG_PATH}/reranking.log
    sleep 1m

    curl http://${ip_address}:8008/generate \
        -X POST \
        -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":64, "do_sample": true}}' \
        -H 'Content-Type: application/json' > ${LOG_PATH}/generate.log
    sleep 5s

    curl http://${ip_address}:9000/v1/chat/completions \
        -X POST \
        -d '{"text":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json' > ${LOG_PATH}/completions.log
    sleep 5s
}

function validate_megaservice() {
    # Curl the Mega Service
    curl http://${ip_address}:8888/v1/chatqna -H "Content-Type: application/json" -d '{
        "model": "Intel/neural-chat-7b-v3-3",
        "messages": "What is the revenue of Nike in 2023?"}' > ${LOG_PATH}/curl_megaservice.log

    echo "Checking response results, make sure the output is reasonable. "
    local status=false
    if [[ -f $LOG_PATH/curl_megaservice.log ]] && \
    [[ $(grep -c "data: 5" $LOG_PATH/curl_megaservice.log) != 0 ]] && \
    [[ $(grep -c "data: 1" $LOG_PATH/curl_megaservice.log) != 0 ]] && \
    [[ $(grep -c "data: ." $LOG_PATH/curl_megaservice.log) != 0 ]] && \
    [[ $(grep -c "data: 2" $LOG_PATH/curl_megaservice.log) != 0 ]]; then
        status=true
    fi

    if [ $status == false ]; then
        echo "Response check failed, please check the logs in artifacts!"
        exit 1
    else
        echo "Response check succeed!"
    fi

    echo "Checking response format, make sure the output format is acceptable for UI."
    # TODO
}

function stop_docker() {
    cd $WORKPATH/microservice/gaudi
    container_list=$(cat docker_compose.yaml | grep container_name | cut -d':' -f2)
    for container_name in $container_list; do
        cid=$(docker ps -aq --filter "name=$container_name")
        if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
    done
}

function main() {

    stop_docker

    build_docker_images
    start_services

    validate_microservices
    validate_megaservice

    stop_docker
    echo y | docker system prune

}

main