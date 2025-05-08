import torch
import numpy as np
from transformers import AutoTokenizer
from generate_outputs import generate
from transformers import AutoModelForCausalLM

def sample_wmh(probs, rng=np.random.default_rng(seed=42)):
    n = probs.shape[0]
    print(f"n: {n}")
    while True:
        u = rng.uniform(0, n)
        j = int(np.floor(u))
        print(f"j: {j}, u: {u}, probs[j]: {probs[j]}")
        if u < j + probs[j]:
            return j 

def sample_gumbel(probs, rng=np.random.default_rng(seed=42)):
    public_random_number_tensor = torch.asarray(rng.uniform(0, 1, size=probs.shape[0])).cuda()
    gumbel_tensor = -torch.log(public_random_number_tensor)/probs
    return torch.argmin(gumbel_tensor).item()

def sample_top1(probs, rng=None):
    return torch.argmax(probs).item()

def compare_output(prompt, desired, sampling_method, length=50):
    print(f"Prompt: {prompt}")
    with torch.no_grad():
        rng = np.random.default_rng()
        ordinary_output = []
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
        for i in range(length):
            ordinary_token, probs = generate(desired, prompt_tokens)
            ordinary_token = sample_top1(probs)
            ordinary_output.append(ordinary_token)
            prompt_tokens = torch.cat([prompt_tokens, torch.tensor([ordinary_token], dtype=prompt_tokens.dtype, device=prompt_tokens.device).unsqueeze(0)], dim=1)
        print(f"Ordinary Output: {tokenizer.decode(ordinary_output, skip_special_tokens=True)}")
        
        sampled_output = []
        for i in range(length):
            _, probs = generate(drafter, prompt_tokens)
            sampled_token = sampling_method(probs, rng)
            sampled_output.append(sampled_token)
            prompt_tokens = torch.cat([prompt_tokens, torch.tensor([sampled_token], dtype=prompt_tokens.dtype, device=prompt_tokens.device).unsqueeze(0)], dim=1)
        print(f"Sampled Output: {tokenizer.decode(sampled_output, skip_special_tokens=True)}")

@torch.inference_mode()
def verify(
    verifier,
    drafter_token_seqs,
    drafter_tokens:      torch.LongTensor,
    drafter_probs:       torch.FloatTensor,
    sampling_method,
    rng
) -> torch.LongTensor:
    """
    Speculative verification (Algorithm 1)
    """
    device = drafter_token_seqs[0].device
    gamma  = drafter_tokens.numel()

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

    verified_tokens = []
    for k, (ver_dist, d_tok, p_tok) in enumerate(zip(verifier_probs, drafter_tokens)):
        verified_token = sampling_method(ver_dist, rng)
        verified_tokens.append(verified_token)
        if verified_token != d_tok:
            break
    return torch.tensor(verified_tokens)

@torch.inference_mode()
def drafter_invariant_speculative_decoding_step(prompt_tokens, drafter, verifier, tokenizer, sampling_method, rng, max_length=50):
    drafter_token_seqs = [prompt_tokens]
    drafter_tokens = []
    drafter_probs = []

    for _ in range(max_length):
        _, probs = generate(drafter, drafter_token_seqs[-1])
        drafter_token = sampling_method(probs, rng)
        drafter_tokens.append(drafter_token)
        drafter_probs.append(probs)
        drafter_token_seqs.append(torch.cat([drafter_token_seqs[-1], torch.tensor([drafter_token], dtype=drafter_token_seqs[-1].dtype, device=drafter_token_seqs[-1].device).unsqueeze(0)], dim=1))
    drafter_outputs = tokenizer.decode(drafter_token_seqs[-1].squeeze(0), skip_special_tokens=True)
    print(f"Drafter Output: {drafter_outputs}")

    drafter_tokens = torch.tensor(drafter_tokens, device=drafter_token_seqs[0].device, dtype=drafter_token_seqs[0].dtype)
    drafter_probs =  torch.stack(drafter_probs, dim=0).to(device=drafter_token_seqs[0].device, dtype=torch.float32)
    
    verified_tokens = verify(
        verifier,
        drafter_token_seqs,
        drafter_tokens,
        drafter_probs,
        sampling_method,
        rng
    )
    verified_output = tokenizer.decode(verified_tokens.tolist(), skip_special_tokens=True)
    return verified_tokens

def drafter_invariant_speculative_decoding(prompt, drafter, verifier, tokenizer, sampling_method, rng, max_length=50, trials=10): 
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
    for _ in range(trials):
        verified_tokens = drafter_invariant_speculative_decoding_step(prompt_tokens, drafter, verifier, tokenizer, sampling_method, rng, max_length=10)
        prompt_tokens = torch.cat([prompt_tokens, verified_tokens.unsqueeze(0)], dim=1)
        prompt_updated = tokenizer.decode(prompt_tokens[0].tolist(), skip_special_tokens=True)
        print(f"Prompt Updated: {prompt_updated}")

prompt = "Write me a poem about Machine Learning."
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
