# Linear_Reasoning

# Hybrid Chain-of-Thought Inference Framework

A research‐grade Python library combining a linear‐time state‐space model (SSM) with an attention‐based Transformer to generate, verify, and finalize chain-of-thought reasoning traces for numeric QA tasks.

---

## 🚀 Motivation

Standard Transformers incur **quadratic compute & memory** when you ask them to “think step by step.” This project offloads the heavy lifting of draft reasoning to a **linear‐time SSM**, then uses a lightweight Transformer to **audit & correct** that draft—delivering accurate answers with significantly lower inference cost.

---

## 🔧 Key Features

- **Two‐Stage “Reversed CoT” Pipeline**  
  1. **Draft Reasoning:** An SSM generates a detailed chain-of-thought trace in linear time.  
  2. **Finalization:** An attention‐based causal Transformer audits, corrects, and outputs the final numeric answer.

- **Supervised Fine-Tuning with LoRA SFT**  
  - Chat-style formatting of CoT examples.  
  - Fine-tuned SSM on 20K reasoning samples using TRL’s `SFTTrainer` + PEFT LoRA (r=16, α=32, dropout=0.05) over 3 epochs (batch=2, grad-accum=4, AdamW @1e-4, 3% warmup, FP16).

- **Comprehensive Compute Profiling**  
  - **`torch.profiler`**: CPU/CUDA FLOPs, shape tracing, peak & end GPU memory  
  - **`psutil`**: CPU utilization  
  - **Token counts & latency** logging per example  
  - Structured JSON reports in `output_llama-1b-base/`

- **Performance Optimizations**  
  - `device_map="auto"` + mixed-precision FP16 for ~2× speed-up and ~30% peak‐GPU savings  
  - Plug-and-play: swap SSM/Transformer backbones via a simple CLI or Python API  
  - Jupyter notebook examples visualize reasoning traces & metrics for fast iteration

---

## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/SAI-ADITH/Linear_Reasoning.git
   cd Linear_Reasoning
