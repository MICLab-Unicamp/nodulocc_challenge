from pathlib import Path
import re
import os
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from dotenv import load_dotenv
load_dotenv()

model_id = "google/medgemma-1.5-4b-it"
adapter_checkpoint = "checkpoints/medgemma-1.5-cls-ckpt"

if adapter_checkpoint is None:
    raise ValueError(f"No checkpoint found in {adapter_checkpoint}")

print("Using adapter checkpoint:", adapter_checkpoint)

merge_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

base_model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=merge_dtype,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_id)

peft_model = PeftModel.from_pretrained(base_model, adapter_checkpoint)
merged_model = peft_model.merge_and_unload()

MERGED_OUT = Path("./medgemma-1.5-4b-it-nodulocc-cls-merged")
MERGED_OUT.mkdir(parents=True, exist_ok=True)

merged_model.save_pretrained(MERGED_OUT, safe_serialization=True, max_shard_size="5GB")
processor.save_pretrained(MERGED_OUT)

print(f"Merged model saved locally to: {MERGED_OUT}")

# Optional push
repo_id = "k298976/medgemma-1.5-4b-it-nodulocc-cls"
merged_model.push_to_hub(repo_id)
processor.push_to_hub(repo_id)
print(f"Pushed merged model to: {repo_id}")