# AI Customer Support Reply Evaluator

## Overview

This project builds an AI-based system for evaluating customer support replies using a rubric-based scoring approach.

The system takes a customer support task and an agent response, then predicts a score from 0 to 4 and generates a short rationale explaining the evaluation. The goal is to make reply evaluation more consistent, faster, and easier to interpret.

The project uses an instruction-tuned transformer model as a baseline and then improves it using LoRA fine-tuning.

---

## Problem Statement

Evaluating customer support replies manually can be time-consuming and inconsistent. Different evaluators may give different scores for the same response, especially when the quality depends on several factors such as empathy, correctness, clarity, and professional tone.

This project addresses the problem by training an AI model to evaluate responses according to a fixed rubric. The model is expected to produce both:

* A numerical score from 0 to 4
* A rationale explaining the reason behind the score

---

## Input and Output

### Input

Each sample contains:

* `task`: the customer request or complaint
* `reference`: an ideal response
* `submission`: the response being evaluated
* `rubric`: the evaluation criteria
* `score`: the ground-truth score
* `rationale`: the human explanation for the score

### Output

The model outputs:

* `score`: predicted score from 0 to 4
* `rationale`: short explanation for the predicted score

---

## Scoring Scale

| Score | Meaning                    |
| ----- | -------------------------- |
| 0     | Unacceptable response      |
| 1     | Very weak response         |
| 2     | Partially helpful response |
| 3     | Good response              |
| 4     | Excellent response         |

---

## Evaluation Rubric

The model evaluates each response based on the following criteria:

1. Shows understanding and apology
2. Provides correct and relevant information
3. Provides a clear solution or next step
4. Uses polite and professional tone

---

## Dataset

The dataset is stored in JSONL format. Each line represents one evaluation sample.

Example:

```json
{
  "task": "Customer complains that their order has not arrived after the expected delivery date.",
  "reference": "We sincerely apologize for the delay. Your order is currently on its way and will arrive within 2 to 3 business days. We understand your frustration and truly appreciate your patience. Please do not hesitate to contact us if you need any further assistance.",
  "submission": "Wait for it. Deliveries take time.",
  "rubric": {
    "1": "Shows understanding and apology",
    "2": "Provides correct and relevant information",
    "3": "Provides clear solution or next step",
    "4": "Uses polite and professional tone"
  },
  "score": 1,
  "rationale": "The reply is dismissive and unhelpful. It vaguely implies the order will arrive but offers no apology, no timeline, and no professional tone."
}
```

The final dataset contains 200 samples and was split into:

* Training set: 160 samples
* Validation set: 20 samples
* Test set: 20 samples

The split was created using stratified sampling to preserve the score distribution across all sets.

---

## Model Approach

### Baseline Model

The baseline model was evaluated before fine-tuning to establish an initial performance reference.

Base model:

* Mistral-7B-Instruct
* 4-bit quantized version for memory efficiency

### Fine-Tuning

LoRA fine-tuning was applied to adapt the model to the rubric-based evaluation task without updating all model parameters.

Several LoRA experiments were tested by changing configuration settings such as:

* LoRA rank
* Learning rate
* Number of epochs
* Target modules

The best-performing configuration was Experiment 5, where LoRA was applied to more target modules:

```text
q_proj, v_proj, k_proj, o_proj
```

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy: exact match between predicted score and true score
* MAE: average distance between predicted and true score
* QWK: agreement between predictions and true scores while considering score distance

### Final Results

| Model                 | Accuracy |  MAE |    QWK   | ROUGE-L | BERTScore |
| --------------------- | -------: | ---: | -------: | -------: | --------: |
| Baseline Model        |   80.00% | 0.20 | 0.9545   | 0.1769   | 0.8943    |
| Fine-Tuned LoRA Model |   90.00% | 0.10 | 0.9744   | 0.3982   | 0.9195    |

The fine-tuned model improved accuracy by 10 percentage points and reduced MAE from 0.20 to 0.10. This shows that LoRA fine-tuning helped the model better align with the rubric-based scoring task.

---

## Project Structure

```text
customer-support-reply-evaluator/
│
├── data/
│   ├── dataset.jsonl
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── baseline_predictions.jsonl
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_finetuning.ipynb
│   └── 04_evaluation.ipynb
│
├── outputs/
│   └── evaluation/
│       └── final_comparison_results.csv
│
├── src/
│   ├── prompts.py
│   ├── extract_prediction.py
│   ├── generate_response.py
│   ├── get_accuracy.py
│   └── inference.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Notebook Workflow

### 1. Data Exploration

The first notebook checks the dataset structure, score distribution, missing values, duplicate samples, and text length statistics. It also creates the train, validation, and test splits.

### 2. Baseline Model

The second notebook evaluates the instruction-tuned model before fine-tuning. It generates predictions on the test set and saves the baseline results.

### 3. Fine-Tuning

The third notebook applies LoRA fine-tuning and compares different experiment configurations. Experiment 5 achieved the best results.

### 4. Final Evaluation

The fourth notebook summarizes and compares the baseline results with the best fine-tuned LoRA experiment. It also saves the final comparison results as a CSV file.

---

## User Interface

A simple user interface is provided to test the evaluator.

The UI allows the user to:

* Enter a customer task
* Enter a response submission
* View the predicted score
* View the generated rationale
* Generate random examples
* Show or hide the reference answer
* View the scoring rubric

---

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Depending on the UI setup, the project may also be run with Streamlit:

```bash
streamlit run app.py
```

---

## Authors

* Amr Asmar
* Nancy Al-Jabbarin

An-Najah National University

---

## Future Work

Possible improvements include:

* Adding CSV export from the UI
* Showing score breakdown per rubric item
* Adding dashboard analytics
* Comparing multiple models inside the interface
* Expanding the dataset with more diverse customer support scenarios
