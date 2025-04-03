import sys
import os

# Add system Python path where packaging is installed
# sys.path.append('/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/python-3.10.1-5r/lib/python3.10/site-packages')

# # Original imports follow
# os.environ['CFLAGS'] = '-DPY_SSIZE_T_CLEAN'
# import ctypes
# ctypes.cdll.LoadLibrary('libc.so.6').prctl(1, 15)

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 
dtype = None 
load_in_4bit = False 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,

)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,
    loftq_config = None, 
)

def get_reasoning_llama_template(tokenizer):

    # Modify the template to include reasoning
    tokenizer.chat_template = '''<|begin_of_text|>{% for message in messages %}{% if message["role"] == "user" %}<|start_header_id|>user<|end_header_id|>

{{ message["content"] }}<|eot_id|>{% elif message["role"] == "reasoning" %}<|start_header_id|>reasoning<|end_header_id|>

{{ message["content"] }}<|eot_id|>{% elif message["role"] == "assistant" %}<|start_header_id|>assistant<|end_header_id|>

{{ message["content"] }}<|eot_id|>{% endif %}{% endfor %}'''

    return tokenizer

tokenizer = get_reasoning_llama_template(tokenizer)

def formatting_prompts_func(examples):
    conversations = []
    for i in range(len(examples['user'])):
        conv = [
            {"role": "user", "content": examples['user'][i]},
            {"role": "reasoning", "content": examples['reasoning'][i]},
            {"role": "assistant", "content": examples['assistant'][i]}
        ]
        conversations.append(conv)

    texts = [tokenizer.apply_chat_template(convo, tokenize=False) for convo in conversations]
    return {"text": texts}

# Load the reasoning dataset
from datasets import load_dataset
dataset = load_dataset("KingNish/reasoning-base-20k", split="train")

# Format the dataset
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

# # Verify the formatting
# # print("\nExample formatted conversation:")
# # print(dataset[5]["text"])
# # print(dataset[6]["text"])


# Training Setup
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# Update trainer to handle reasoning outputs
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>reasoning<|end_header_id|>\n\n",
)

# Training Execution #
trainer_stats = trainer.train()

# # Save Model #
model.save_pretrained("llama32_1b_reasoning") # Local saving
tokenizer.save_pretrained("llama32_1b_reasoning")

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")