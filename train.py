# train.py - Clean, production-ready training script
import argparse
from datasets import load_dataset
from transformers import (
    GPTNeoForCausalLM, GPT2TokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from config import get_tinystories_config

parser = argparse.ArgumentParser(description="Train TinyStories 10M/28M/33M models")
parser.add_argument("--model_size", type=str, default="28M", choices=["10M", "28M", "33M"])
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--push_to_hub", action="store_true")
args = parser.parse_args()

# Model & tokenizer
config = get_tinystories_config(args.model_size)
model = GPTNeoForCausalLM(config)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Dataset
dataset = load_dataset("roneneldan/TinyStories")

def tokenize_function(examples):
    return tokenizer(examples["text"], add_special_tokens=False)

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

output_dir = args.output_dir or f"TinyStories-{args.model_size}"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=5e-4,
    weight_decay=0.1,
    lr_scheduler_type="constant",
    num_train_epochs=args.epochs,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=True,
    report_to=[],
    push_to_hub=args.push_to_hub,
    hub_model_id=f"yourname/TinyStories-{args.model_size}-reproduction" if args.push_to_hub else None,
    dataloader_num_workers=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

print(f"Training TinyStories-{args.model_size}...")
trainer.train()
trainer.save_model()
print(f"Done! Model saved to {output_dir}")
