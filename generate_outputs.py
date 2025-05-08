import torch

def generate(model, input_ids):
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1).item()
    return next_token_id, probs.squeeze(0)