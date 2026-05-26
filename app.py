import gradio as gr
import random
from src.prompts import build_prompt
from src.extract_prediction import extract_prediction
from src.generate_response import generate_response
from src.inference import load_model

model, tokenizer, device = load_model()

RUBRIC = {
    "1": "Understands the customer’s problem clearly",
    "2": "Shows empathy and acknowledges frustration",
    "3": "Provides correct and relevant solution",
    "4": "Uses polite and professional tone"
}


SCENARIOS = [
    # ===== Score 4 =====
    {
        "task": "Customer did not receive their order after the expected delivery date and is requesting urgent update.",
        "reference": "We sincerely apologize for the delay. Your order is on the way and will arrive within 2–3 business days.",
        "submission": "We sincerely apologize for the delay and understand your frustration. Your order is currently on the way and will arrive within 2–3 business days. Please let us know if you need further assistance.",
    },

    # ===== Score 3 =====
    {
        "task": "Customer received a damaged product and wants refund or replacement.",
        "reference": "We apologize and will offer a replacement or refund immediately.",
        "submission": "We are sorry for the damaged product. We can offer a refund or replacement. Please tell us your choice.",
    },

    # ===== Score 2 =====
    {
        "task": "Customer complains about delayed delivery.",
        "reference": "We apologize for the delay and will update the customer with delivery time.",
        "submission": "Your order is delayed. It will arrive soon.",
    },

    # ===== Score 1 =====
    {
        "task": "Customer reports missing item in order.",
        "reference": "We apologize and will investigate and resend missing item.",
        "submission": "Check your order again.",
    },

    # ===== Score 0 =====
    {
        "task": "Customer complains about double charge on account.",
        "reference": "We will verify the transaction and refund any duplicate charge.",
        "submission": "Not our problem. Contact your bank.",
    },

    # ===== More realistic variations =====
    {
        "task": "Customer asks for refund after receiving wrong item.",
        "reference": "We apologize and will issue refund or replacement immediately.",
        "submission": "We will check and get back to you soon.",
    }
]



current_reference = {"text": "", "visible": False}

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
    output = generate_response(model, tokenizer, prompt, device)
    parsed = extract_prediction(output)

    return parsed.get("score"), parsed.get("rationale")


# ======================
# UI
# ======================
with gr.Blocks(theme=gr.themes.Soft(), title="AI Evaluator") as demo:

    gr.Markdown("# 🤖 Customer Support Reply Evaluator")

    # =========================
    # TOP SECTION (INPUT)
    # =========================
    with gr.Column():
        task = gr.Textbox(label="Task", lines=3)
        submission = gr.Textbox(label="Submission", lines=3)

        with gr.Row():
            eval_btn = gr.Button("🚀 Evaluate", variant="primary")
            gen_btn = gr.Button("🎲 Generate Example")

        score = gr.Number(label="Score (0–4)")
        rationale = gr.Textbox(label="Rationale", lines=5)

    # =========================
    # BOTTOM SECTION (DETAILS)
    # =========================
    gr.Markdown("---")
    gr.Markdown("## 📌 Evaluation Details")

    # ---- Rubric nice UI ----
    with gr.Accordion("📊 Rubric (Evaluation Criteria)", open=False):

        for k, v in RUBRIC.items():
            gr.Markdown(f"""
### Criterion {k}
{v}
""")

    # ---- Reference ----
    with gr.Accordion("📄 Reference Answer (Hidden by default)", open=False):

        ref_box = gr.Textbox(label="Reference", interactive=False)
        ref_state = gr.Textbox(value="🔒 Hidden", label="Status", interactive=False)

        toggle_btn = gr.Button("👁 Toggle Reference")

    # =========================
    # EVENTS
    # =========================
    eval_btn.click(
        fn=predict,
        inputs=[task, submission],
        outputs=[score, rationale]
    )

    gen_btn.click(
        fn=generate_example,
        inputs=[],
        outputs=[task, submission, ref_box, ref_state]
    )

    toggle_btn.click(
        fn=toggle_reference,
        inputs=[],
        outputs=[ref_box, ref_state]
    )

demo.launch()