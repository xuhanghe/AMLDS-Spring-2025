from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def generate(model, input_ids):
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1).item()
    return next_token_id, probs.squeeze(0)

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
        print(" n (first rejection index):", first_fail)
        print(" Drafter token probabilities: p =", float(p_token[first_fail]), 
            ", q =", float(q_token[first_fail]), 
            ", ratio =", float(ratio[first_fail]), 
            ", r =", float(r[first_fail]))
        print(" Rejected drafter token: ",
            tokenizer.decode(int(drafter_tokens[first_fail])))
    else:
        n = gamma
    print(f"n: {n}")

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
def speculative_decoding_step(prompt_tokens, drafter, verifier, tokenizer, max_length=50):
    drafter_token_seqs = [prompt_tokens]
    drafter_tokens = []
    drafter_probs = []

    for _ in range(max_length):
        drafter_token, probs = generate(drafter, drafter_token_seqs[-1])
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
        drafter_probs 
    )
    verified_output = tokenizer.decode(verified_tokens.tolist(), skip_special_tokens=True)
    return verified_tokens

def speculative_decoding(prompt, drafter, verifier, tokenizer): 
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
    for _ in range(10):
        verified_tokens = speculative_decoding_step(prompt_tokens, drafter, verifier, tokenizer, max_length=10)
        prompt_tokens = torch.cat([prompt_tokens, verified_tokens.unsqueeze(0)], dim=1)
        prompt_updated = tokenizer.decode(prompt_tokens[0].tolist(), skip_special_tokens=True)
        print(f"Prompt Updated: {prompt_updated}")

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

def sample_top1(probs):
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

prompt = "Write me a poem about Machine Learning."
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

drafter = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# verifier = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2-27b-it",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

compare_output(prompt, drafter, sample_gumbel)





