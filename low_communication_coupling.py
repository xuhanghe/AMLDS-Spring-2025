from transformers import AutoTokenizer
from generate_outputs import generate
from transformers import AutoModelForCausalLM
import torch
import numpy as np

def low_communication_coupling_step(drafter, desired, prompt_tokens, rng):
    with torch.no_grad():
        drafter_token, drafter_probs = generate(drafter, prompt_tokens)
        _, desired_probs = generate(desired, prompt_tokens)

        u_0 = rng.uniform(0, 1)
        if u_0 < min(desired_probs[drafter_token]/drafter_probs[drafter_token], 1):
            return drafter_token

        desired_cumsum = torch.cumsum(desired_probs, dim=-1)
        while True:
            u_k = float(rng.uniform(0,1))
            j = int(torch.searchsorted(desired_cumsum, torch.tensor([u_k]), right=False).item())
            w = float(desired_cumsum[j-1]) if j>0 else 0.0

            if float(drafter_probs[j]) < (u_k - w):
                return j

