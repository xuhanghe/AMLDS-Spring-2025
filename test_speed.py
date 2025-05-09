from communication_free_coupling import drafter_invariant_speculative_decoding_step, sample_gumbel
from speculative_decoding import speculative_decoding_step
from low_communication_coupling import low_communication_coupling_step
from generate_outputs import auto_regression_step
from experiments.Timer import Timer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import json
import numpy as np
import random

"""
Note: Due to implementation differences, the time measured here is just for eye-balling the performances

HC3 dataset structure:
    {"question": ...,
    "human_answers":[...],
    "chatgpt_answers":[...],
    "index": ...,
    "source": ...
    }, ...
    python test_speed.py \
  --HC3_dataset '/scratch/xh2433/amlds_final_project/HC3_datasets/open_qa.jsonl' \
  --method speculative_decoding \
  --output_length 10 \
  --num_drafter_tokens_per_step 5 \
|& tee outputs.txt

"""



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="generate the test of speed of different sampling methods")
    parser.add_argument("--HC3_dataset", type=str, required=True, help="Input path to the HC3 dataset")
    parser.add_argument("--method", type=str, required=True, help="which method to test")
    parser.add_argument("--output_length", type=int, default=16, help="length of output tokens")
    parser.add_argument("--num_drafter_tokens_per_step", type=int, default=5, help="number of drafter tokens per step")

    return parser.parse_args()

def main(args):
    with open(args.HC3_dataset, "r") as f:
        data = [json.loads(line.strip()) for line in f]

    data = data[:500]
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
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

    if args.method == "speculative_decoding":
        step_func = speculative_decoding_step
    elif args.method == "drafter_invariant_speculative_decoding":
        step_func = drafter_invariant_speculative_decoding_step
    elif args.method == "low_communication_coupling":
        step_func = low_communication_coupling_step
    elif args.method == "auto-regression":
        step_func = auto_regression_step
    else:
        raise NotImplementedError(f"Method {args.method} not implemented")
    
    sampling_method = sample_gumbel
    TTLT = []
    Avg_token_level_latency = []
    Avg_tokens_per_step = []
    results = {}
    print("before enumerate")
    
    for i, item in enumerate(data):
        print(f"Processing item {i+1}/{len(data)}")
        prompt = item["question"]
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
        num_output_tokens = 0
        num_steps = 0
        token_level_timer = Timer()
        rng = np.random.default_rng(seed=random.randint(0, 3000000))
        while True:
            token_level_timer.start()
            verified_tokens = step_func(
                prompt_tokens=prompt_tokens, 
                drafter=drafter, 
                verifier=verifier, 
                tokenizer=tokenizer, 
                sampling_method=sampling_method,
                rng=rng,
                max_length=args.num_drafter_tokens_per_step
            )
            print(f"Verified tokens: {verified_tokens}")
            token_level_timer.stop()
            num_output_tokens += len(verified_tokens)
            num_steps += 1
            if num_output_tokens >= args.output_length:
                token_level_timer.num_tokens = num_output_tokens
                break
            prompt_tokens = torch.cat([prompt_tokens, torch.tensor(verified_tokens).unsqueeze(0).cuda()], dim=1)
            prompt_updated = tokenizer.decode(prompt_tokens[0].tolist(), skip_special_tokens=True)
        Avg_token_level_latency.append(token_level_timer.get_average_duration())
        TTLT.append(token_level_timer.get_total_duration())
        Avg_tokens_per_step.append(num_output_tokens / num_steps)
        results[i] = {"Avg_token_level_latency": np.mean(Avg_token_level_latency),
                           "Avg_tokens_per_step": np.mean(Avg_tokens_per_step),
                           "Total_time_taken": np.sum(TTLT)}
        if i != 0 and i % 10 == 0:
            print(f"Avg token level latency: {np.mean(Avg_token_level_latency)}")
            print(f"Avg tokens per step: {np.mean(Avg_tokens_per_step)}")
            print(f"Total time taken: {np.sum(TTLT)}")
            with open(f"{args.method}_test_speed_results.json", "w") as f:
                json.dump(results, f, indent=4)


    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)