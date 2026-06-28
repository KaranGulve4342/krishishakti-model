#!/bin/bash
set -e

# Download model files from HF bucket if not already present
NEED_DOWNLOAD=false
[ ! -f "/app/yield_model.pkl" ] && NEED_DOWNLOAD=true
[ ! -f "/app/crop_model.pkl" ] && NEED_DOWNLOAD=true
[ ! -f "/app/plant_disease_model.pth" ] && NEED_DOWNLOAD=true

if [ "$NEED_DOWNLOAD" = true ]; then
    echo "Downloading model files from Hugging Face bucket..."
    hf sync hf://buckets/KaranGulve/KrishishaktiModels /tmp/hf_bucket

    [ ! -f "/app/yield_model.pkl" ] && mv /tmp/hf_bucket/yield_model.pkl /app/yield_model.pkl && echo "yield_model.pkl ready."
    [ ! -f "/app/crop_model.pkl" ] && mv /tmp/hf_bucket/crop_model.pkl /app/crop_model.pkl && echo "crop_model.pkl ready."
    [ ! -f "/app/plant_disease_model.pth" ] && mv /tmp/hf_bucket/plant_disease_model.pth /app/plant_disease_model.pth && echo "plant_disease_model.pth ready."

    rm -rf /tmp/hf_bucket
fi

exec uvicorn app:app --host 0.0.0.0 --port 7860
