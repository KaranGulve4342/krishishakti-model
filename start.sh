#!/bin/bash
set -e

if [ ! -f "/app/yield_model.pkl" ]; then
    echo "Downloading yield_model.pkl from Hugging Face..."
    python3 -c "
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
fs.get('hf://buckets/KaranGulve/KrishishaktiModels/yield_model.pkl', '/app/yield_model.pkl')
print('Download complete.')
"
fi

exec uvicorn app:app --host 0.0.0.0 --port 8080
