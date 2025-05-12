import torch
import numpy as np
import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate_outputs import generate

def quantized_gumbel_coupling_step(prompt_tokens, drafter, verifier, rng, bit_budget):
    """
    Algorithm 6: b-Bit Quantized Gumbel Coupling
    Returns (drafter_token, verifier_token).
    """
    with torch.no_grad():
        _, drafter_probs = generate(drafter, prompt_tokens)
        _, verifier_probs = generate(verifier, prompt_tokens)
        drafter_probs = drafter_probs.cpu().numpy()
        verifier_probs = verifier_probs.cpu().numpy()

        # Shared uniform random value and quantization
        U = float(rng.uniform(0, 1))
        k = int(U * (2 ** bit_budget))
        U_tilde = k / (2 ** bit_budget)

        # Drafter chooses based on verifier_probs
        scores_d = -np.log(U) / verifier_probs
        drafter_token = int(np.argmin(scores_d))

        # Verifier chooses based on drafter_probs and quantized U
        scores_v = -np.log(U_tilde) / drafter_probs
        verifier_token = int(np.argmin(scores_v))

        return drafter_token, verifier_token

@torch.inference_mode()
def b_bit_speculative_decoding_step(prompt_tokens, drafter, verifier, rng, bit_budget, gamma):
    """
    Algorithm 7: γ-Step b-Bit Speculative Decoding.
    Returns the next accepted token as a list.
    """
    prefix = prompt_tokens

    # Draft & verify γ steps
    for _ in range(gamma):
        a, b = quantized_gumbel_coupling_step(prefix, drafter, verifier, rng, bit_budget)
        if a != b:
            return [b]
        # extend prefix with matched draft
        prefix = torch.cat([
            prefix,
            torch.tensor([[a]], device=prefix.device, dtype=prefix.dtype)
        ], dim=1)

    # If all γ matched, sample one more from true verifier distribution
    _, probs = generate(verifier, prefix)
    probs = probs.cpu()
    cumsum = torch.cumsum(probs, dim=-1)
    u = float(rng.uniform(0, 1))
    next_token = int(torch.searchsorted(cumsum, torch.tensor([u]), right=False).item())
    return [next_token]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    drafter = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it", device_map="auto", torch_dtype=torch.bfloat16
    )
    verifier = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-27b-it", device_map="auto", torch_dtype=torch.bfloat16
    )

    # Load data (first 500 items)
    with open("/scratch/xh2433/amlds_final_project/HC3_datasets/open_qa.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    data = data[:500]

    result = {}
    bit_budget = 8
    gamma = 5

    for i, item in enumerate(data):
        print(f"Processing item {i+1}/{len(data)}")
        prompt = item["question"]
        prompt_tokens = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to("cuda").input_ids

        generated_answers = []
        for _ in range(3):
            rng = np.random.default_rng(seed=random.randint(0, 3_000_000))
            while True:
                verified_tokens = b_bit_speculative_decoding_step(
                    prompt_tokens, drafter, verifier, rng, bit_budget, gamma
                )
                prompt_tokens = torch.cat([
                    prompt_tokens,
                    torch.tensor(verified_tokens, device=prompt_tokens.device).unsqueeze(0)
                ], dim=1)
                prompt_updated = tokenizer.decode(
                    prompt_tokens[0].tolist(), skip_special_tokens=True
                )
                print(f"Prompt Updated: {prompt_updated}")
                if prompt_tokens[0, -1].item() == tokenizer.eos_token_id:
                    break
            generated_answers.append(prompt_updated)

        result[i] = {
            "question": prompt,
            "answer": generated_answers,
            "human_answers": item.get("human_answers", [""])[0]
        }

        # Periodic save
        if (i + 1) % 20 == 0:
            with open("quantized_gumbel_coupling_results.json", "w") as out_f:
                json.dump(result, out_f, indent=4)

