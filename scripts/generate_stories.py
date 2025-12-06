# scripts/generate_stories.py
import torch
from transformers import GPTNeoForCausalLM, GPT2TokenizerFast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--prompt", type=str, default="Once upon a time, ")
parser.add_argument("--num_stories", type=int, default=5)
args = parser.parse_args()

model = GPTNeoForCausalLM.from_pretrained(args.checkpoint)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model.eval()

for i in range(args.num_stories):
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=300,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        eos_token_id=50256,
        pad_token_id=50256
    )
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n=== Story {i+1} ===\n{story}\n")
