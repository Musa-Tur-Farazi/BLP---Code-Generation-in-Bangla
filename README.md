### Bangla Language Processing (BLP) Workshop — Task 2: Bangla-to-Code System

This repo contains a minimal, end-to-end system to train and run a Bangla instruction-to-Python code generator that produces `submission.json` in the exact required format.

It includes:
- Fine-tuning script for causal LMs with LoRA
- Data loader for the BLP dataset JSON
- Prompt formatter tailored for Bangla instructions
- Inference script to generate `submission.json`
- Simple dev evaluator to run provided asserts locally


## Environment setup

- OS: Windows 10+ supported
- Python: 3.9–3.11 recommended

Create and activate a virtual environment, then install dependencies.

```powershell
# From the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

If you have a CUDA GPU and want faster training, install a CUDA-enabled PyTorch before `requirements.txt`:

```powershell
# Example for CUDA 12.1 (adjust to your setup)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```


## Data format

The dataset JSON is a list of objects, each with fields:
- `id`: integer unique id
- `instruction`: Bangla natural language prompt
- `response`: Python function definition (training only; absent at eval)
- `test_list`: a Python list literal encoded as a string with assert statements for the function (dev only)

Example entry:
```json
{
  "id": 231,
  "instruction": "প্রদত্ত অ্যারে থেকে সমান উপাদান জোড়া গণনা করার জন্য একটি পাইথন ফাংশন লিখুন।",
  "response": "def count_Pairs(arr,n):\r\n    cnt = 0; \r\n    for i in range(n): \r\n        for j in range(i + 1,n): \r\n            if (arr[i] == arr[j]): \r\n                cnt += 1; \r\n    return cnt; ",
  "test_list": "['assert count_Pairs([1,1,1,1],4) == 6', 'assert count_Pairs([1,5,1],3) == 1', 'assert count_Pairs([3,2,1,7,8,9],6) == 0']"
}
```


## Training

The trainer fine-tunes a base code model using supervised learning with masked loss so only the target function is learned. LoRA reduces memory usage.

Minimal example:
```powershell
python -m src.training ^
  --train_json path\to\train.json ^
  --eval_json  path\to\dev.json ^
  --output_dir outputs\blp25_lora ^
  --model_name bigcode/tiny_starcoder_py ^
  --num_train_epochs 3 ^
  --per_device_train_batch_size 2 ^
  --learning_rate 2e-4 ^
  --lora_r 16 ^
  --lora_alpha 32 ^
  --lora_dropout 0.05
```

Notes:
- Replace `bigcode/tiny_starcoder_py` with a larger base model if you have the resources, e.g. `bigcode/starcoderbase-1b` or `codellama/CodeLlama-7b-Python-hf`.
- On CPU-only setups, keep epochs small and use the tiny model.
- You can resume training by pointing `--output_dir` to the same path.


## Generating submission.json

Given a JSON with only `id` and `instruction` (evaluation input), run:
```powershell
python -m src.generate ^
  --input_json path\to\eval_input.json ^
  --output_json submission.json ^
  --base_model bigcode/tiny_starcoder_py ^
  --adapter_dir outputs\blp25_lora ^
  --max_new_tokens 512 ^
  --temperature 0.2 ^
  --top_p 0.95
```

This writes a file with the exact format:
```json
[
  {"id": 231, "response": "def ..."},
  {"id": 999, "response": "def ..."}
]
```


## Dev evaluation (optional)

You can locally check whether generated functions pass the provided asserts in the dev split:
```powershell
python -m src.evaluate_dev ^
  --data_json path\to\dev.json ^
  --pred_json submission.json
```
This executes each predicted function with the asserts in `test_list` and prints per-item pass/fail counts. This is a convenience tool only; the official evaluation runs programs in a sandbox.


## Tips for better results
- Keep the prompt template consistent between training and inference. The code uses a concise Bangla template that explicitly asks for only a Python function definition.
- Constrain generation to prefer valid Python by using low temperature and setting `eos_token` to stop early. The script already configures common EOS tokens.
- Use a code-specialized base model for best results.


## Reproducibility

Both training and generation accept `--seed` to set RNG seeds for deterministic behavior where possible.


## File overview
- `src/data_utils.py`: load and validate JSON data
- `src/prompting.py`: prompt templates and formatting
- `src/training.py`: LoRA fine-tuning entry point
- `src/generate.py`: inference entry point producing `submission.json`
- `src/evaluate_dev.py`: simple assert-based checker for dev data


## License
This scaffold uses permissive, widely used open-source libraries. Check individual model licenses when selecting a base model.