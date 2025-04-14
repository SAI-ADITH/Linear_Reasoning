from datasets import load_dataset
from trl import SFTTrainer, SFTConfig  # Changed import
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b-hf",
    trust_remote_code=True,
    fused_add_norm=True,  # Required for fast path
    device_map="cuda",
)

model.config.use_mambapy = True

print("Fast path enabled:", model.config.use_mambapy)
print(model.config)

# Add special tokens and template (same as before)
tokenizer.chat_template = """<|begin_of_text|>
{% for message in messages %}
{% if message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'reasoning' %}<|start_header_id|>reasoning<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>
{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}"""
additional_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})
model.resize_token_embeddings(len(tokenizer))

# Dataset formatting (same as before)
def format_dataset(examples):
    formatted = []
    for user, reasoning, assistant in zip(examples['user'], examples['reasoning'], examples['assistant']):
        formatted.append(tokenizer.apply_chat_template([
            {"role": "user", "content": user},
            {"role": "reasoning", "content": reasoning},
            {"role": "assistant", "content": assistant}
        ], tokenize=False))
    return {"text": formatted}

dataset = load_dataset("KingNish/reasoning-base-20k", split="train")
dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names)

# Use SFTConfig instead of TrainingArguments
training_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,
    optim="adamw_torch",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    max_seq_length=2048,
    dataset_text_field="text",  # Now in SFTConfig
    packing=True,
)

# LoRA config remains the same
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none",
    lora_dropout=0.05
)

# Initialize trainer with SFTConfig
trainer = SFTTrainer(
    model=model,
    args=training_config,  # Use SFTConfig here
    peft_config=lora_config,
    train_dataset=dataset,
)

# Training and saving remains the same
trainer.train()
model.save_pretrained("mamba_2.8b_reasoning")
tokenizer.save_pretrained("mamba_2.8b_reasoning")   