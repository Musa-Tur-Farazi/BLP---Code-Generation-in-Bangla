#!/usr/bin/env python3
"""
Official Task Approach for BLP25 Code Generation
Adapts our research model to work with the official task format.
"""

import argparse
import json
import pandas as pd
import torch
import re
from tqdm.auto import tqdm
from typing import List, Dict, Any

from src.research_model import BanglaCodeResearchModel, BanglaCodeResearchConfig
from src.data_utils import load_dataset, BlpItem
from src.prompting import build_prompt


def create_official_prompt(instruction: str) -> str:
    """Create prompt in official task format"""
    return f"""You are a Python programming expert. Write a Python function based on the following instruction in Bangla.

Instruction: {instruction}

Write only the Python function code, wrapped in ```python code blocks. Do not include any explanations or comments outside the code block."""


def extract_function_from_response(response: str) -> str:
    """Extract function code from model response"""
    # Remove markdown code blocks
    if response.startswith("```python"):
        response = response[9:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    
    # Clean up whitespace
    response = response.strip()
    
    # Ensure it's a function
    if not response.startswith("def "):
        # Try to find function definition
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                response = '\n'.join(lines[i:])
                break
    
    return response


def validate_response(response: str) -> bool:
    """Validate if response is properly formatted"""
    # Check if it's wrapped in code fences
    fence_pattern = re.compile(r"^```python[\s\S]*```$", re.MULTILINE)
    if not fence_pattern.match(response):
        return False
    
    # Check if it contains a function definition
    if "def " not in response:
        return False
    
    return True


def generate_official_submission(
    model_path: str,
    dev_csv_path: str,
    output_json: str = "submission.json",
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.95,
    top_k: int = 64
) -> None:
    """Generate submission in official format"""
    
    # Load model
    print("Loading research model...")
    config = BanglaCodeResearchConfig()
    model = BanglaCodeResearchModel.from_pretrained(model_path, config=config)
    
    # Load tokenizer (use a standard one for compatibility)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Load dev data
    print("Loading dev dataset...")
    df = pd.read_csv(dev_csv_path)
    assert {"id", "instruction"}.issubset(df.columns), "CSV must have columns: id, instruction"
    
    print(f"Loaded {len(df)} samples from {dev_csv_path}")
    
    # Generate responses
    print("Generating code...")
    responses = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        instruction = str(row["instruction"])
        
        # Create prompt
        prompt = create_official_prompt(instruction)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)
        
        # Create instruction mask
        seq_length = inputs["input_ids"].shape[1]
        instruction_length = int(seq_length * 0.8)
        instruction_mask = torch.zeros((1, seq_length), dtype=torch.bool, device=device)
        instruction_mask[:, :instruction_length] = True
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                instruction_mask=instruction_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract only the generated part
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        response = generated_text[len(prompt_text):].strip()
        
        # Ensure proper formatting
        if not response.startswith("```python"):
            response = f"```python\n{response}\n```"
        
        # Validate and fix if needed
        if not validate_response(response):
            # Try to fix common issues
            if "def " in response and not response.startswith("```python"):
                response = f"```python\n{response}\n```"
            elif not "def " in response:
                # Generate a basic function template
                response = f"```python\ndef solution():\n    # TODO: Implement based on instruction\n    pass\n```"
        
        responses.append(response)
    
    # Create submission DataFrame
    out_df = pd.DataFrame({
        "id": df["id"],
        "response": responses
    })
    
    # Save as JSON
    out_df.to_json(output_json, orient="records", force_ascii=False, indent=2)
    print(f"✅ Wrote {output_json} with {len(out_df)} rows")
    
    # Validate format
    validate_submission_format(output_json)


def validate_submission_format(json_path: str) -> bool:
    """Validate submission format according to official requirements"""
    
    def file_format_check(path: str) -> bool:
        # Check file name
        if not path.endswith(".json"):
            print("Error: File must have .json extension")
            return False
        
        # Load and validate JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return False
        
        if not isinstance(data, list):
            print("Error: Root element should be a list")
            return False
        
        # Validate each item
        fence_pat = re.compile(r"^```python[\s\S]*```$", re.MULTILINE)
        
        valid_format = 0
        valid_fence = 0
        valid_both = 0
        
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"Error: Item {idx} is not a dictionary")
                return False
            
            keys = set(item.keys())
            if keys != {"id", "response"}:
                print(f"Error: Item {idx} must have only 'id' and 'response' keys")
                return False
            
            if not isinstance(item["id"], int):
                print(f"Error: 'id' at index {idx} must be integer")
                return False
            
            if not isinstance(item["response"], str):
                print(f"Error: 'response' at index {idx} must be string")
                return False
            
            valid_format += 1
            
            # Check code fencing
            if fence_pat.match(item["response"]):
                valid_fence += 1
                valid_both += 1
        
        print(f"Format valid: {valid_format}/{len(data)} ({valid_format*100.0/len(data):.1f}%)")
        print(f"Fencing valid: {valid_fence}/{len(data)} ({valid_fence*100.0/len(data):.1f}%)")
        print(f"Both valid: {valid_both}/{len(data)} ({valid_both*100.0/len(data):.1f}%)")
        
        return True
    
    return file_format_check(json_path)


def create_submission_zip(json_path: str, zip_path: str = "submission.zip") -> None:
    """Create submission zip file"""
    import zipfile
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path)
    
    print(f"📦 Created {zip_path} containing {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate official submission using research model")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained research model")
    parser.add_argument("--dev_csv", type=str, required=True, help="Path to dev_v2.csv")
    
    # Optional arguments
    parser.add_argument("--output_json", type=str, default="submission.json", help="Output JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens")
    parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=64, help="Top-k sampling")
    parser.add_argument("--create_zip", action="store_true", help="Create submission.zip")
    
    args = parser.parse_args()
    
    # Generate submission
    generate_official_submission(
        model_path=args.model_path,
        dev_csv_path=args.dev_csv,
        output_json=args.output_json,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Create zip if requested
    if args.create_zip:
        create_submission_zip(args.output_json)


if __name__ == "__main__":
    main()
