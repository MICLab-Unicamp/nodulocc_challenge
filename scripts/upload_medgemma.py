from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import torch
from dotenv import load_dotenv
load_dotenv()

model_id = "google/medgemma-1.5-4b-it"
adapter_repo = "k298976/medgemma-1.5-4b-it-nodulocc-cls-qlora-adapter"

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)

adapter_model = PeftModel.from_pretrained(model, "checkpoints/medgemma-1.5-nodulocc-cls-ckpt")

adapter_model.push_to_hub(adapter_repo)
processor.push_to_hub(adapter_repo)