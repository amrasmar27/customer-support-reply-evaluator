import torch

def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response