# 🤖 AI Customer Support Reply Evaluator

## 📌 Overview

This project is an AI-powered system that automatically evaluates customer support responses using an instruction-tuned transformer model (Mistral-7B-Instruct fine-tuned with LoRA).

The system takes structured input:
- Task
- Reference answer (hidden in UI)
- Submission
- Rubric

It outputs:
- Score (0–4)
- Rationale (explanation)

---

## 🎯 Problem Statement

Manual evaluation of customer support replies is:
- Time-consuming
- Inconsistent
- Subjective

This project solves this using an AI model that follows a fixed rubric for consistent and explainable scoring.

---

## 🧠 Model

- Base Model: Mistral-7B-Instruct (4-bit)
- Fine-tuning: LoRA (Unsloth)
- Frameworks:
  - HuggingFace Transformers
  - PEFT
  - Unsloth

---

## 📊 Scoring

| Score | Meaning |
|------|--------|
| 0 | Unacceptable |
| 1 | Very weak |
| 2 | Partially helpful |
| 3 | Good |
| 4 | Excellent |

---

## 📋 Rubric

- Understanding the customer’s problem
- Empathy and acknowledgment
- Correct solution
- Professional tone

---

## 📁 Dataset Format

```jsonl
{
"task":"Customer complains that their order has not arrived after the expected delivery date.",
"reference":"We sincerely apologize for the delay. Your order is currently on its way and will arrive within 2 to 3 business days. We understand your frustration and truly appreciate your patience. Please do not hesitate to contact us if you need any further assistance.",
"submission":"Wait for it. Deliveries take time.",
"rubric":{"1":"Shows understanding and apology","2":"Provides correct and relevant information","3":"Provides clear solution or next step","4":"Uses polite and professional tone"},
"score":1,
"rationale":"The reply is dismissive and unhelpful. It vaguely implies the order will arrive but offers no apology, no timeline, and no professional tone whatsoever."}
```

---

## 📂 Project Structure

```
customer-support-reply-evaluator/
│
├── src/
├── data/
├── notebooks/
├── models/
├── reports/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🖥️ UI Features

- Task input
- Submission input
- Score output
- Rationale explanation
- Random examples generator
- Reference toggle (hidden by default)
- Static rubric display

---

## ⚙️ Workflow

1. Input task + submission
2. Build prompt
3. Run model
4. Extract score + rationale
5. Display results

---

## 📈 Evaluation

Metrics:
- QWK (Quadratic Weighted Kappa)
- Accuracy
- MAE

Comparison:
- Base model vs Fine-tuned model

---

## 🚀 Run

```bash
pip install -r requirements.txt
python app.py
```

---

## 👨‍💻 Authors

- Amr Asmar  
- Nancy Al-Jabbarin  
An-Najah National University

---

## ⭐ Future Work

- CSV export
- Score breakdown per rubric item
- Dashboard analytics
- Model comparison UI
