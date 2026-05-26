import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model():

    MODEL_PATH = "models/best_lora_model"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return model, tokenizer, device