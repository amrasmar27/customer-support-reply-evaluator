import gradio as gr
import random
import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import build_prompt
from src.extract_prediction import extract_prediction
from src.generate_response import generate_response
from src.inference import load_model


# ======================
# LOAD MODELS
# ======================

# Fine-tuned model (LoRA)
finetuned_model, finetuned_tokenizer, device = load_model()

# Baseline model (Mistral base)
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

baseline_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

baseline_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)


# ======================
# RUBRIC
# ======================
RUBRIC = {
    "1": "Understands the customer’s problem clearly",
    "2": "Shows empathy and acknowledges frustration",
    "3": "Provides correct and relevant solution",
    "4": "Uses polite and professional tone"
}


# ======================
# SCENARIOS
# ======================
SCENARIOS = [
    {
        "task": "Customer did not receive their order after delivery date.",
        "reference": "We sincerely apologize. Your order will arrive in 2–3 days.",
        "submission": "We sincerely apologize for the delay. Your order is on the way."
    },
    {
        "task": "Customer received damaged product.",
        "reference": "We will refund or replace immediately.",
        "submission": "We are sorry. We can refund or replace."
    },
    {
        "task": "Customer complains about delayed delivery.",
        "reference": "We apologize and will update delivery time.",
        "submission": "Your order is delayed."
    },
    {
        "task": "Customer reports missing item.",
        "reference": "We will resend missing item.",
        "submission": "We will check and get back to you."
    },
    {
        "task": "Customer reports double charge.",
        "reference": "We will refund duplicate charge.",
        "submission": "Not our problem. Contact bank."
    }
]


# ======================
# STATE
# ======================
current_reference = {"text": "", "visible": False}


# ======================
# FUNCTIONS
# ======================
def generate_example():
    ex = random.choice(SCENARIOS)
    current_reference["text"] = ex["reference"]
    current_reference["visible"] = False
    return ex["task"], ex["submission"], "", "🔒 Hidden"


def toggle_reference():
    current_reference["visible"] = not current_reference["visible"]

    if current_reference["visible"]:
        return current_reference["text"], "🔓 Visible"
    return "", "🔒 Hidden"


def predict(task, submission):

    row = {
        "task": task,
        "reference": current_reference["text"],
        "submission": submission,
        "rubric": RUBRIC
    }

    prompt = build_prompt(row)

    # ======================
    # BASELINE
    # ======================
    baseline_output = generate_response(
        baseline_model,
        baseline_tokenizer,
        prompt,
        device
    )
    baseline_parsed = extract_prediction(baseline_output)

    # ======================
    # FINE-TUNED
    # ======================
    finetuned_output = generate_response(
        finetuned_model,
        finetuned_tokenizer,
        prompt,
        device
    )
    finetuned_parsed = extract_prediction(finetuned_output)

    return (
        baseline_parsed.get("score"),
        baseline_parsed.get("rationale"),
        finetuned_parsed.get("score"),
        finetuned_parsed.get("rationale")
    )


# ======================
# METRICS DATA
# ======================
metrics_df = pd.DataFrame([
    {
        "Model": "Baseline Model",
        "Accuracy": 0.80,
        "MAE": 0.20,
        "QWK": 0.954545,
        "ROUGE-L": 0.176876,
        "BERTScore": 0.894274
    },
    {
        "Model": "Fine-Tuned LoRA (Exp5)",
        "Accuracy": 0.90,
        "MAE": 0.10,
        "QWK": 0.974400,
        "ROUGE-L": 0.398176,
        "BERTScore": 0.919493
    }
])


# ======================
# UI
# ======================
with gr.Blocks(theme=gr.themes.Soft(), title="AI Evaluator") as demo:

    gr.Markdown("# 🤖 Customer Support Reply Evaluator")

    with gr.Tabs():

        # ======================
        # TAB 1: EVALUATOR
        # ======================
        with gr.Tab("🧪 Evaluator"):

            task = gr.Textbox(label="Task", lines=3)
            submission = gr.Textbox(label="Submission", lines=3)

            with gr.Row():
                eval_btn = gr.Button("🚀 Evaluate", variant="primary")
                gen_btn = gr.Button("🎲 Generate Example")

            gr.Markdown("## 📊 Model Comparison")

            with gr.Row():

                with gr.Column():
                    gr.Markdown("### 📉 Baseline Model")
                    baseline_score = gr.Number(label="Score (0–4)")
                    baseline_rationale = gr.Textbox(label="Rationale", lines=5)

                with gr.Column():
                    gr.Markdown("### 🚀 Fine-tuned Model")
                    finetuned_score = gr.Number(label="Score (0–4)")
                    finetuned_rationale = gr.Textbox(label="Rationale", lines=5)

            # events inside tab
            eval_btn.click(
                fn=predict,
                inputs=[task, submission],
                outputs=[
                    baseline_score,
                    baseline_rationale,
                    finetuned_score,
                    finetuned_rationale
                ]
            )

            gen_btn.click(
                fn=generate_example,
                inputs=[],
                outputs=[task, submission]
            )

        # ======================
        # TAB 2: METRICS TABLE
        # ======================
        with gr.Tab("📊 Metrics Table"):

            gr.Markdown("## Model Performance Comparison")

            gr.Dataframe(
                value=metrics_df,
                interactive=False
            )

    # ======================
    # OPTIONAL: REFERENCE
    # ======================
    with gr.Accordion("📄 Reference Answer", open=False):
        ref_box = gr.Textbox(label="Reference", interactive=False)
        ref_state = gr.Textbox(value="🔒 Hidden", label="Status", interactive=False)
        toggle_btn = gr.Button("👁 Toggle Reference")

        toggle_btn.click(
            fn=toggle_reference,
            inputs=[],
            outputs=[ref_box, ref_state]
        )


demo.launch()