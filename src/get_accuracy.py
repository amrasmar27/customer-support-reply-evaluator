import torch
import json
import re
import numpy as np
from sklearn.metrics import ( mean_absolute_error,cohen_kappa_score)
from src.prompts import build_prompt

def get_accuracy(model, tokenizer, test_df, device, num_samples=None):
    model.eval()

    df = test_df if num_samples is None else test_df.head(num_samples)

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        prompt = build_prompt(row)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        pred_score = None
        try:
            pred_score = int(json.loads(generated)["score"])
        except:
            match = re.search(r'"score"\s*:\s*(\d)', generated)
            if match:
                pred_score = int(match.group(1))

        true_score = int(row["score"])

        if pred_score is not None:
            y_true.append(true_score)
            y_pred.append(pred_score)

    acc = np.mean(np.array(y_true) == np.array(y_pred)) if len(y_true) > 0 else 0
    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"QWK:      {qwk:.4f}")

    return acc, y_true, y_pred