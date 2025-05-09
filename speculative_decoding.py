from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from generate_outputs import generate
import json

@torch.inference_mode()
def verify(
    verifier,
    drafter_token_seqs,
    drafter_tokens:      torch.LongTensor,
    drafter_probs:       torch.FloatTensor,
) -> torch.LongTensor:
    """
    Speculative verification (Algorithm 1)
    """
    device = drafter_token_seqs[0].device
    gamma  = drafter_tokens.numel()

    if isinstance(drafter_probs, list):
        drafter_probs = torch.stack(drafter_probs, dim=0)
    drafter_probs = drafter_probs.to(device=device, dtype=torch.float32)

    if isinstance(drafter_tokens, list):
        drafter_tokens = torch.tensor(drafter_tokens, device=device, dtype=torch.long)
    else:
        drafter_tokens = drafter_tokens.to(device=device, dtype=torch.long)

    with torch.no_grad():
        lengths = torch.tensor(
            [seq.size(1) for seq in drafter_token_seqs],
            device=device,
            dtype=torch.long,
        )
        pads = [seq.squeeze(0) for seq in drafter_token_seqs]
        padded = pad_sequence(pads, batch_first=True, padding_value=verifier.config.pad_token_id)

        logits = verifier(input_ids=padded).logits
        last_logits = logits[torch.arange(len(lengths), device=device), lengths - 1]
        verifier_probs = F.softmax(last_logits, dim=-1)

    p_token = verifier_probs[:gamma].gather(1, drafter_tokens[:, None]).squeeze(1)
    q_token = drafter_probs.gather(1, drafter_tokens.unsqueeze(1)).squeeze(1)

    ratio = p_token / torch.clamp_min(q_token, 1e-30)
    r     = torch.rand_like(ratio)

    fail_mask = r > ratio
    if fail_mask.any():
        n = int(fail_mask.nonzero(as_tuple=True)[0][0])
        first_fail = int(fail_mask.nonzero(as_tuple=True)[0][0])
        # print(" n (first rejection index):", first_fail)
        # print(" Drafter token probabilities: p =", float(p_token[first_fail]), 
        #     ", q =", float(q_token[first_fail]), 
        #     ", ratio =", float(ratio[first_fail]), 
        #     ", r =", float(r[first_fail]))
        # print(" Rejected drafter token: ",
        #     tokenizer.decode(int(drafter_tokens[first_fail])))
    else:
        n = gamma
    # print(f"n: {n}")

    p_next = verifier_probs[n]
    if n < gamma:
        diff = torch.clamp_min(p_next - drafter_probs[n], 0.0)
        mass = diff.sum()
        p_prime = p_next if mass < 1e-8 else diff / mass
    else:
        p_prime = p_next

    t = torch.multinomial(p_prime, 1)

    verified_tokens = torch.cat([drafter_tokens[:n], t])

    return verified_tokens

@torch.inference_mode()
def speculative_decoding_step(prompt_tokens, drafter, verifier, tokenizer, rng=None, sampling_method=None, max_length=10):
    drafter_token_seqs = [prompt_tokens]
    drafter_tokens = []
    drafter_probs = []

    with torch.no_grad():
        for _ in range(max_length):
            drafter_token, probs = generate(drafter, drafter_token_seqs[-1])
            drafter_tokens.append(drafter_token)
            drafter_probs.append(probs)
            drafter_token_seqs.append(torch.cat([drafter_token_seqs[-1], torch.tensor([drafter_token], dtype=drafter_token_seqs[-1].dtype, device=drafter_token_seqs[-1].device).unsqueeze(0)], dim=1))
        drafter_outputs = tokenizer.decode(drafter_token_seqs[-1].squeeze(0), skip_special_tokens=True)
    # print(f"Drafter Output: {drafter_outputs}")

    drafter_tokens = torch.tensor(drafter_tokens, device=drafter_token_seqs[0].device, dtype=drafter_token_seqs[0].dtype)
    drafter_probs =  torch.stack(drafter_probs, dim=0).to(device=drafter_token_seqs[0].device, dtype=torch.float32)
    
    verified_tokens = verify(
        verifier,
        drafter_token_seqs,
        drafter_tokens,
        drafter_probs 
    )
    # verified_output = tokenizer.decode(verified_tokens.tolist(), skip_special_tokens=True)
    return verified_tokens.tolist()

def speculative_decoding(prompt, drafter, verifier, tokenizer): 
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
    for _ in range(10):
        verified_tokens = speculative_decoding_step(prompt_tokens, drafter, verifier, tokenizer, max_length=10)
        prompt_tokens = torch.cat([prompt_tokens, torch.tensor(verified_tokens).cuda().unsqueeze(0)], dim=1)
        prompt_updated = tokenizer.decode(prompt_tokens[0].tolist(), skip_special_tokens=True)
        # print(f"Prompt Updated: {prompt_updated}")


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
                verified_tokens = speculative_decoding_step(prompt_tokens, drafter, verifier, tokenizer, max_length=10)
                prompt_tokens = torch.cat([prompt_tokens, torch.tensor(verified_tokens).cuda().unsqueeze(0)], dim=1)
                prompt_updated = tokenizer.decode(prompt_tokens[0].tolist(), skip_special_tokens=True)
                print(f"Prompt Updated: {prompt_updated}")
                last_token_id = prompt_tokens[0, -1].item()
                if last_token_id == tokenizer.eos_token_id:
                    break
                            
            generated_answers.append(prompt_updated)
        result[i] = {"question": prompt, "answer": generated_answers, "human_answers": item["human_answers"][0]}
        
        if (i + 1) % 20 == 0:
            with open("speculative_decoding_results.json", "w") as f:
                json.dump(result, f, indent=4)
    
    







