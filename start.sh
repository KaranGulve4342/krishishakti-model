#!/bin/bash
set -e

REQUIRED_FILES=("yield_model.pkl" "crop_model.pkl" "plant_disease_model.pth")
MISSING=false

for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "/app/$f" ]; then
        MISSING=true
        break
    fi
done

if [ "$MISSING" = true ]; then
    echo "Downloading model files from Hugging Face bucket..."
    hf sync hf://buckets/KaranGulve/KrishishaktiModels /tmp/hf_bucket

    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "/app/$f" ]; then
            if [ ! -f "/tmp/hf_bucket/$f" ]; then
                echo "ERROR: $f not found in HF bucket. Upload it first."
                exit 1
            fi
            mv "/tmp/hf_bucket/$f" "/app/$f"
            echo "$f ready."
        fi
    done

    rm -rf /tmp/hf_bucket
fi

exec uvicorn app:app --host 0.0.0.0 --port 7860
