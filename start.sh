#!/bin/bash
set -e

if [ ! -f "/app/yield_model.pkl" ]; then
    echo "Downloading yield_model.pkl from Hugging Face bucket..."
    hf sync hf://buckets/KaranGulve/KrishishaktiModels /tmp/hf_bucket
    mv /tmp/hf_bucket/yield_model.pkl /app/yield_model.pkl
    rm -rf /tmp/hf_bucket
    echo "Download complete."
fi

exec uvicorn app:app --host 0.0.0.0 --port 7860
