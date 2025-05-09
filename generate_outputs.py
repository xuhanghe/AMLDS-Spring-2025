import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import json
import random


def generate(model, input_ids):
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1).item()
    return next_token_id, probs.squeeze(0)

def auto_regression_step(prompt_tokens, verifier, rng=None, drafter=None, tokenizer=None, sampling_method=None, max_length=None):
    next_token, _ = generate(verifier, prompt_tokens)
    return [next_token]


if __name__ == "__main__":
    # prompt = "Write me a poem about Machine Learning."
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

    drafter = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    verifier = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-27b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    with open("/scratch/xh2433/amlds_final_project/HC3_datasets/open_qa.jsonl", "r") as f:
        data = [json.loads(line.strip()) for line in f]
    data = data[:500]

    result = {}

    for i, item in enumerate(data):
        print(f"Processing item {i+1}/{len(data)}")
        prompt = item["question"]
        prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda").input_ids

        generated_answers = []
        for _ in range(3):
            while True:
                with torch.no_grad():
                    verified_tokens = auto_regression_step(prompt_tokens=prompt_tokens, verifier=verifier , max_length=10)
                    # import pdb; pdb.set_trace()
                    prompt_tokens = torch.cat([prompt_tokens, torch.tensor(verified_tokens).cuda().unsqueeze(0)], dim=1)
                    prompt_updated = tokenizer.decode(prompt_tokens[0].tolist(), skip_special_tokens=True)
                    print(f"Prompt Updated: {prompt_updated}")
                    last_token_id = prompt_tokens[0, -1].item()
                    if last_token_id == tokenizer.eos_token_id:
                        break
                            
            generated_answers.append(prompt_updated)
        result[i] = {"question": prompt, "answer": generated_answers, "human_answers": item["human_answers"][0]}
    
    with open("auto_regression_results.json", "w") as f:
        json.dump(result, f, indent=4)