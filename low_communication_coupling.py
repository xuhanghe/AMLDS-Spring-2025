from transformers import AutoTokenizer
from generate_outputs import generate
from transformers import AutoModelForCausalLM
import torch
import numpy as np
import json
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import random

def low_communication_coupling_step(prompt_tokens, drafter, verifier, rng, tokenizer=None, sampling_method=None, max_length=None):
    with torch.no_grad():
        drafter_token, drafter_probs = generate(drafter, prompt_tokens)
        _, verifier_probs = generate(verifier, prompt_tokens)
        verifier_probs = verifier_probs.cpu()
        print(f"Verifier probs shape: {verifier_probs.shape}")

        u_0 = rng.uniform(0, 1)
        if u_0 < min(verifier_probs[drafter_token]/drafter_probs[drafter_token], 1):
            return [drafter_token]

        verifier_cumsum = torch.cumsum(verifier_probs, dim=-1)
        while True:
            u_k = float(rng.uniform(0,1))
            j = int(torch.searchsorted(verifier_cumsum, torch.tensor([u_k]), right=False).item())
            w = float(verifier_cumsum[j-1]) if j>0 else 0.0

            if float(drafter_probs[j]) < (u_k - w):
                return [j]

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
            rng = np.random.default_rng(seed=random.randint(0, 3000000))

            while True:
                verified_tokens = low_communication_coupling_step(prompt_tokens=prompt_tokens, drafter=drafter, verifier=verifier, max_length=10, rng=rng)
                prompt_tokens = torch.cat([prompt_tokens, torch.tensor(verified_tokens).cuda().unsqueeze(0)], dim=1)
                prompt_updated = tokenizer.decode(prompt_tokens[0].tolist(), skip_special_tokens=True)
                print(f"Prompt Updated: {prompt_updated}")
                last_token_id = prompt_tokens[0, -1].item()
                if last_token_id == tokenizer.eos_token_id:
                    break
                            
            generated_answers.append(prompt_updated)
        result[i] = {"question": prompt, "answer": generated_answers, "human_answers": item["human_answers"][0]}
    
        # Save the results to a JSON file 
        if (i + 1) % 20 == 0:
            with open("low_communication_coupling_results.json", "w") as f:
                json.dump(result, f, indent=4)