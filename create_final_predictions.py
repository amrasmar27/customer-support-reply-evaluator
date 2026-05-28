import json
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import build_prompt


BASE_MODEL_NAME = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
LORA_MODEL_PATH = "models/best_lora_model"

TEST_PATH = Path("data/test.jsonl")
BASELINE_PATH = Path("data/baseline_predictions.jsonl")

OUTPUT_CSV = Path("data/final_predictions.csv")
OUTPUT_JSONL = Path("data/final_predictions.jsonl")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def extract_json_prediction(text):
    """
    Extract the score and rationale from the model output.
    The expected output format is JSON with score and rationale.
    """

    try:
        start = text.rfind("{")
        end = text.rfind("}") + 1

        if start != -1 and end > start:
            json_text = text[start:end]
            parsed = json.loads(json_text)

            return {
                "score": int(parsed.get("score")),
                "rationale": str(parsed.get("rationale", ""))
            }

    except Exception:
        pass

    score_match = re.search(r'"?score"?\s*:\s*([0-4])', text)
    rationale_match = re.search(r'"?rationale"?\s*:\s*"([^"]+)"', text)

    score = int(score_match.group(1)) if score_match else None
    rationale = rationale_match.group(1) if rationale_match else ""

    return {
        "score": score,
        "rationale": rationale
    }


def generate_model_output(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


print("Loading test data and baseline predictions...")
test_df = load_jsonl(TEST_PATH)
baseline_df = load_jsonl(BASELINE_PATH)

print("Test samples:", len(test_df))
print("Baseline predictions:", len(baseline_df))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Loading fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
model.eval()

records = []

print("Generating fine-tuned predictions on test set...")

for i, row in test_df.iterrows():
    prompt = build_prompt(row) + "\n\n### Response:\n"

    raw_output = generate_model_output(model, tokenizer, prompt)
    parsed_output = extract_json_prediction(raw_output)

    baseline_row = baseline_df.iloc[i]

    records.append({
        "sample_id": i,
        "task": row["task"],
        "reference": row["reference"],
        "submission": row["submission"],
        "true_score": int(row["score"]),
        "baseline_pred_score": int(baseline_row["pred_score"]),
        "fine_tuned_pred_score": parsed_output["score"],
        "baseline_rationale": baseline_row["rationale"],
        "fine_tuned_rationale": parsed_output["rationale"],
        "raw_fine_tuned_output": raw_output
    })

    print(f"Finished sample {i + 1}/{len(test_df)}")


final_predictions_df = pd.DataFrame(records)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

final_predictions_df.to_csv(OUTPUT_CSV, index=False)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


valid_predictions_df = final_predictions_df.dropna(
    subset=["fine_tuned_pred_score"]
).copy()

valid_predictions_df["fine_tuned_pred_score"] = valid_predictions_df[
    "fine_tuned_pred_score"
].astype(int)

fine_tuned_accuracy = accuracy_score(
    valid_predictions_df["true_score"],
    valid_predictions_df["fine_tuned_pred_score"]
)

fine_tuned_mae = mean_absolute_error(
    valid_predictions_df["true_score"],
    valid_predictions_df["fine_tuned_pred_score"]
)

fine_tuned_qwk = cohen_kappa_score(
    valid_predictions_df["true_score"],
    valid_predictions_df["fine_tuned_pred_score"],
    weights="quadratic"
)

print("\nFine-Tuned Detailed Prediction Metrics")
print(f"Accuracy: {fine_tuned_accuracy * 100:.2f}%")
print(f"MAE: {fine_tuned_mae:.2f}")
print(f"QWK: {fine_tuned_qwk:.4f}")

print("\nSaved detailed predictions files:")
print(OUTPUT_CSV)
print(OUTPUT_JSONL)