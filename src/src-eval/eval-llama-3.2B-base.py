# import os
# import random
# import re
# import json
# import time
# from datasets import load_dataset
# from vllm import LLM, SamplingParams

# ds = load_dataset("openai/gsm8k", "main", split="test")

# random.seed(24)

# num_samples = 10
# sample_indices = random.sample(range(len(ds)), num_samples)

# baseline_prompt = "Answer the question and give the final answer as a number and end your response with '###END###': "

# def extract_numbers(text):
#     if text:
#         text = text.replace(",", "")
#         numbers = re.findall(r'-?\d+\.?\d*', text)
#         return [float(num) for num in numbers]
#     return []

# # model_path = "/scratch/ssenthi2/llama-r/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
# model_path = "/scratch/ssenthi2/llama-r/llama32_1b_reasoning"

# llm = LLM(
#             model=model_path,
#             dtype="float16",
#             tensor_parallel_size=1,
#             max_model_len=4096,
#             )

# sampling_params = SamplingParams(
#     max_tokens=4096,
#     temperature=0.7,
#     stop=["###END###"]
# )

# def get_response(prompts):
#     try:
#         outputs = llm.generate(prompts, sampling_params=sampling_params)
#         return outputs
#     except Exception as e:
#         print(f"Error during generation: {e}")
#         return None

# max_retries_for_batch = 5
# max_retries_for_individual = 3

# output_dir = "./llama-3.2-1B-reasoning"
# os.makedirs(output_dir, exist_ok=True)

# for run in range(1, 2):

#     prompts = []
#     for idx in sample_indices:
#         question = ds[idx]["question"]
#         full_prompt = baseline_prompt + question
#         prompts.append(full_prompt)

#     responses = get_response(prompts)

#     if responses is None:
#         for attempt in range(max_retries_for_batch):
#             time.sleep(2)  
#             responses = get_response(prompts)
#             if responses is not None:
#                 break

#     if responses is None:

#         print(f"Failed to get responses after multiple batch retries for run {run}.")
#         continue

#     for i, output in enumerate(responses):
#         generated_text = (output.outputs[0].text if output.outputs else "").strip()

#         if not generated_text:

#             prompt_to_retry = [prompts[i]]
#             single_response = None
#             for attempt in range(max_retries_for_individual):
#                 time.sleep(1)  
#                 single_response = get_response(prompt_to_retry)
#                 if single_response and single_response[0].outputs and single_response[0].outputs[0].text.strip():

#                     responses[i] = single_response[0]
#                     break
#             else:

#                 print(f"Failed to get a valid response for prompt index {i} after multiple individual retries.")

#     results = []
#     correct_count = 0

#     for i, output in enumerate(responses):
#         generated_text = (output.outputs[0].text if output.outputs else "").strip()
#         correct_ans = extract_numbers(ds[sample_indices[i]]["answer"])
#         pred_ans = extract_numbers(generated_text)

#         match = pred_ans and correct_ans and (
#             pred_ans[-1] == correct_ans[-1] or
#             (len(pred_ans) > 1 and pred_ans[-2] == correct_ans[-1]) or
#             (len(pred_ans) > 2 and pred_ans[-3] == correct_ans[-1]) or
#             (len(pred_ans) > 3 and pred_ans[-4] == correct_ans[-1]) or
#             (len(pred_ans) > 4 and pred_ans[-5] == correct_ans[-1])
#         )

#         if match:
#             correct_count += 1

#         results.append({
#             "ID": sample_indices[i],
#             "exact_prompt": prompts[i],
#             "LLM_output": generated_text,
#             "extracted_answer": pred_ans[-1] if pred_ans else "N/A",
#             "correct_answer": correct_ans[-1] if correct_ans else "N/A",
#             "evaluation": "Correct" if match else "Incorrect"
#         })

#     score_summary = {
#         "Final_Score": f"{correct_count}/{num_samples}",
#         "Total_Correct": correct_count,
#         "Total_Questions": num_samples
#     }
#     results.append(score_summary)

#     output_file = os.path.join(output_dir, f"results_run_{run}.json")
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=4)

#     print(f"Results for run {run} saved to {output_file}")
#     print(f"Final Score for run {run}: {correct_count}/{num_samples}\n")

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset
# import torch
# import os
# import random
# import re
# import json
# import time

# # -------------------------------
# # 1. Configuration and Model Loading
# # -------------------------------
# MAX_OUTPUT_TOKENS = 2048
# model_name = "llama32_1b_reasoning"

# # Load model and tokenizer using transformers
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype=torch.float16, device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # -------------------------------
# # 2. Load GSM8K Dataset and Sample Prompts
# # -------------------------------
# ds = load_dataset("openai/gsm8k", "main", split="test")
# random.seed(24)
# num_samples = 10
# sample_indices = random.sample(range(len(ds)), num_samples)

# baseline_prompt = (
#     "Answer the question and give the final answer as a number and end your response with '###END###': "
# )

# def extract_numbers(text):
#     """Extracts numbers from text and returns them as a list of floats."""
#     if text:
#         text = text.replace(",", "")
#         numbers = re.findall(r'-?\d+\.?\d*', text)
#         return [float(num) for num in numbers]
#     return []

# # -------------------------------
# # 3. Generation Helper Function
# # -------------------------------
# def generate_response(prompts):
#     """
#     Tokenizes a list of prompts and generates responses using the model.
#     Returns the raw generated token IDs.
#     """
#     inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=MAX_OUTPUT_TOKENS,
#         temperature=0.7,
#         do_sample=True
#     )
#     return outputs

# # -------------------------------
# # 4. Setup Output Directory and Retry Parameters
# # -------------------------------
# output_dir = "./llama-3.2-1B-reasoning"
# os.makedirs(output_dir, exist_ok=True)

# max_retries_for_batch = 5
# max_retries_for_individual = 3

# # -------------------------------
# # 5. Prepare Prompts for Evaluation
# # -------------------------------
# prompts = []
# for idx in sample_indices:
#     question = ds[idx]["question"]
#     full_prompt = baseline_prompt + question
#     prompts.append(full_prompt)

# # -------------------------------
# # 6. Generate Responses in Batch with Retries
# # -------------------------------
# outputs = None
# for attempt in range(max_retries_for_batch):
#     try:
#         outputs = generate_response(prompts)
#         break
#     except Exception as e:
#         print(f"Batch generation failed on attempt {attempt+1} with error: {e}")
#         time.sleep(2)

# if outputs is None:
#     print("Failed to generate responses for the batch after multiple retries.")
#     exit(1)

# # Decode batch outputs
# responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# # Post-process responses: trim output at the stop token "###END###"
# processed_responses = []
# for response in responses:
#     end_idx = response.find("###END###")
#     if end_idx != -1:
#         response = response[:end_idx]
#     processed_responses.append(response.strip())

# # If any response is empty, retry generating that prompt individually
# for i, response in enumerate(processed_responses):
#     if not response:
#         prompt_retry = prompts[i]
#         single_response = None
#         for attempt in range(max_retries_for_individual):
#             try:
#                 single_output = generate_response([prompt_retry])
#                 single_decoded = tokenizer.decode(single_output[0], skip_special_tokens=True)
#                 end_idx = single_decoded.find("###END###")
#                 if end_idx != -1:
#                     single_decoded = single_decoded[:end_idx]
#                 single_response = single_decoded.strip()
#                 if single_response:
#                     processed_responses[i] = single_response
#                     break
#             except Exception as e:
#                 print(f"Individual retry {attempt+1} failed for prompt index {i} with error: {e}")
#                 time.sleep(1)
#         if not single_response:
#             print(f"Failed to get a valid response for prompt index {i} after multiple retries.")

# # -------------------------------
# # 7. Evaluate and Save the Results
# # -------------------------------
# results = []
# correct_count = 0

# for i, response in enumerate(processed_responses):
#     correct_ans = extract_numbers(ds[sample_indices[i]]["answer"])
#     pred_ans = extract_numbers(response)
    
#     # Check if one of the last few extracted numbers matches the correct answer
#     match = (
#         pred_ans
#         and correct_ans
#         and (
#             pred_ans[-1] == correct_ans[-1]
#             or (len(pred_ans) > 1 and pred_ans[-2] == correct_ans[-1])
#             or (len(pred_ans) > 2 and pred_ans[-3] == correct_ans[-1])
#             or (len(pred_ans) > 3 and pred_ans[-4] == correct_ans[-1])
#             or (len(pred_ans) > 4 and pred_ans[-5] == correct_ans[-1])
#         )
#     )
#     if match:
#         correct_count += 1

#     results.append({
#         "ID": sample_indices[i],
#         "exact_prompt": prompts[i],
#         "LLM_output": response,
#         "extracted_answer": pred_ans[-1] if pred_ans else "N/A",
#         "correct_answer": correct_ans[-1] if correct_ans else "N/A",
#         "evaluation": "Correct" if match else "Incorrect"
#     })

# score_summary = {
#     "Final_Score": f"{correct_count}/{num_samples}",
#     "Total_Correct": correct_count,
#     "Total_Questions": num_samples
# }
# results.append(score_summary)

# output_file = os.path.join(output_dir, "results_run_1.json")
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)

# print(f"Results saved to {output_file}")
# print(f"Final Score: {correct_count}/{num_samples}")


import os
import json
import time
from vllm import LLM, SamplingParams
import re

# === Paths and Configs ===
input_json_path = "prompt.json"
output_json_path = "base_output_1.json"
model_path = "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
num_questions_to_run = 20

# === Sampling Parameters ===
sampling_params = SamplingParams(
    max_tokens=2048,
    temperature=0.7,
)

# === Initialize LLM ===
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    max_model_len=2048,
)

# === Load prompts ===
with open(input_json_path, "r") as f:
    prompts_data = json.load(f)

def extract_numbers(text):
    if text:
        text = text.replace(",", "")
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(num) for num in numbers]
    return []

# === Instruction Prefix ===
instruction_prefix = (
    "Answer the question and give the final answer as a number in the last sentence, "
    "following this exact format: 'Final Answer: {number} \\n####'\n\nQuestion: "
)

# === Evaluation of Model Responses ===
results = []
correct_count = 0

for item in prompts_data[:num_questions_to_run]:
    question = item["question"]
    correct_answer = item["correct_answer"]

    # Generate prompt with instruction
    full_prompt = instruction_prefix + question

    try:
        # Generate model response
        outputs = llm.generate([full_prompt], sampling_params=sampling_params)
        response_text = outputs[0].outputs[0].text.strip() if outputs[0].outputs else "ERROR: No output"
    except Exception as e:
        response_text = f"ERROR: {e}"

    # Extract the model's answer and the correct answer for evaluation
    pred_ans = extract_numbers(response_text)
    correct_ans = extract_numbers(correct_answer)

    # Check if the predicted answer matches the correct answer
    match = pred_ans and correct_ans and (pred_ans[-1] == correct_ans[-1])

    if match:
        correct_count += 1

    # Append result with evaluation
    results.append({
        "question": question,
        "correct_answer": correct_answer,
        "model_output": response_text,
        "extracted_answer": pred_ans[-1] if pred_ans else "N/A",
        "evaluation": "Correct" if match else "Incorrect"
    })

# === Save Results ===
score_summary = {
    "Final_Score": f"{correct_count}/{num_questions_to_run}",
    "Total_Correct": correct_count,
    "Total_Questions": num_questions_to_run
}
results.append(score_summary)

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n✅ Done! Evaluation results saved to {output_json_path}\n")