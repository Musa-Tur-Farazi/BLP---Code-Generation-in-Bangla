# 🎯 Official BLP25 Task Guide with Research Model

## **📋 Task Overview**

**Task**: Bangla-to-Python Code Generation  
**Evaluation**: Pass@1 (percentage of test cases that pass)  
**Format**: Generate Python functions from Bangla instructions  
**Deadline**: September 29, 2025

## **📊 Dataset Structure**

### **Files:**
- **Trial Dataset**: `trial.csv` (training data with solutions)
- **Dev Dataset**: `dev_v2.csv` (development data, no solutions)
- **Test Dataset**: To be released

### **Data Format:**
```csv
id,instruction,test_list
1,"একটি ফাংশন লিখুন যা...","['assert function_name(...) == expected', ...]"
```

### **Expected Output:**
```json
[
  {"id": 1, "response": "```python\ndef function_name(...):\n    # code here\n```"},
  {"id": 2, "response": "```python\ndef another_function(...):\n    # code here\n```"}
]
```

## **🚀 Complete Workflow**

### **Step 1: Environment Setup**
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Data Preparation**
```bash
# Convert trial.csv to JSON for training
python -m src.convert_csv_to_json --input_csv trial.csv --output_json data/train.json

# Verify dev_v2.csv format
python -c "import pandas as pd; df=pd.read_csv('dev_v2.csv'); print(f'Dev samples: {len(df)}'); print(df.head())"
```

### **Step 3: Train Research Model**
```bash
# Train small model for quick experimentation
python -m src.research_training `
  --train_json data/train.json `
  --model_size small `
  --use_dual_stream `
  --use_code_aware_attention `
  --num_epochs 10 `
  --batch_size 4 `
  --learning_rate 1e-4 `
  --output_dir outputs/research_small `
  --experiment_name "official-task-small"

# Train medium model for better performance
python -m src.research_training `
  --train_json data/train.json `
  --model_size medium `
  --use_dual_stream `
  --use_code_aware_attention `
  --num_epochs 15 `
  --batch_size 2 `
  --learning_rate 5e-5 `
  --output_dir outputs/research_medium `
  --experiment_name "official-task-medium"
```

### **Step 4: Generate Official Submission**
```bash
# Generate submission using research model
python -m src.official_approach `
  --model_path outputs/research_medium/latest_model `
  --dev_csv dev_v2.csv `
  --output_json submission.json `
  --temperature 0.3 `
  --max_new_tokens 256 `
  --create_zip
```

### **Step 5: Submit to CodaBench**
1. Upload `submission.zip` to CodaBench dev phase
2. Check leaderboard for real-time results
3. Iterate and improve based on performance

## **🔬 Research Model Advantages**

### **1. Dual-Stream Architecture**
- **Instruction Stream**: Specialized for Bangla natural language
- **Code Stream**: Dedicated for Python function generation
- **Cross-Attention**: Bidirectional information flow

### **2. Code-Aware Attention**
- **Syntax Awareness**: Understands Python structure
- **Function Detection**: Identifies function patterns
- **Bangla-Code Alignment**: Bridges language gap

### **3. Task-Specific Features**
- **Function Generation**: Optimized for Python functions
- **Code Fencing**: Ensures proper ```python blocks
- **Validation**: Automatic format checking

## **📈 Performance Optimization**

### **Model Size Selection:**
```bash
# Quick prototyping (fast training)
python -m src.research_training --model_size tiny --num_epochs 5

# Development (balanced)
python -m src.research_training --model_size small --num_epochs 10

# Production (best performance)
python -m src.research_training --model_size medium --num_epochs 15
```

### **Generation Parameters:**
```bash
# Conservative (higher pass rate)
python -m src.official_approach --temperature 0.1 --top_p 0.9

# Balanced (default)
python -m src.official_approach --temperature 0.3 --top_p 0.95

# Creative (more diverse)
python -m src.official_approach --temperature 0.7 --top_p 0.98
```

## **🔍 Evaluation and Debugging**

### **Local Validation:**
```bash
# Test with ground truth
python -m src.official_approach `
  --model_path outputs/research_medium/latest_model `
  --dev_csv trial.csv `
  --output_json test_submission.json

# Evaluate locally
python -m src.evaluate_dev `
  --submission_json test_submission.json `
  --dev_json data/train.json
```

### **Format Validation:**
```bash
# Check submission format
python -c "
import json
with open('submission.json', 'r') as f:
    data = json.load(f)
print(f'Total items: {len(data)}')
print(f'Sample: {data[0]}')
"
```

## **📊 Expected Performance**

### **Target Metrics:**
- **Pass@1**: 60-80% (vs 10-20% baseline)
- **Format Compliance**: >95%
- **Function Generation**: >90%

### **Research Contributions:**
- **Novel Architecture**: First dual-stream for Bangla-to-code
- **Language-Specific**: Optimized for Bangla instructions
- **Code-Aware**: Understands Python syntax and structure

## **🔄 Iterative Improvement**

### **Phase 1: Baseline**
```bash
# Quick baseline with tiny model
python -m src.research_training --model_size tiny --num_epochs 3
python -m src.official_approach --model_path outputs/tiny_model/latest_model --dev_csv dev_v2.csv
```

### **Phase 2: Optimization**
```bash
# Improve with larger model
python -m src.research_training --model_size small --num_epochs 10
python -m src.official_approach --model_path outputs/small_model/latest_model --dev_csv dev_v2.csv
```

### **Phase 3: Fine-tuning**
```bash
# Best performance with medium model
python -m src.research_training --model_size medium --num_epochs 15
python -m src.official_approach --model_path outputs/medium_model/latest_model --dev_csv dev_v2.csv
```

## **📝 Submission Checklist**

### **Before Submission:**
- [ ] Model trained successfully
- [ ] `submission.json` generated
- [ ] Format validation passed
- [ ] `submission.zip` created
- [ ] File size < 100MB

### **Submission Format:**
```json
[
  {
    "id": 1,
    "response": "```python\ndef function_name(...):\n    # implementation\n    return result\n```"
  }
]
```

### **Validation Rules:**
- ✅ JSON format with list of objects
- ✅ Each object has `id` (int) and `response` (string)
- ✅ Response wrapped in ```python blocks
- ✅ Contains valid Python function

## **🎯 Research vs Official Approach**

### **Research Model Benefits:**
1. **Custom Architecture**: Designed specifically for Bangla-to-code
2. **Dual-Stream**: Better instruction understanding
3. **Code-Aware**: Understands Python syntax
4. **Efficient**: Smaller models with better performance

### **Official Compatibility:**
1. **Same Output Format**: Compatible with official evaluation
2. **Function Generation**: Optimized for Python functions
3. **Code Fencing**: Automatic ```python wrapping
4. **Validation**: Built-in format checking

## **🚀 Quick Start Commands**

### **Complete Pipeline:**
```bash
# 1. Setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. Prepare data
python -m src.convert_csv_to_json --input_csv trial.csv --output_json data/train.json

# 3. Train model
python -m src.research_training `
  --train_json data/train.json `
  --model_size small `
  --use_dual_stream `
  --num_epochs 10 `
  --output_dir outputs/research_model

# 4. Generate submission
python -m src.official_approach `
  --model_path outputs/research_model/latest_model `
  --dev_csv dev_v2.csv `
  --create_zip

# 5. Submit submission.zip to CodaBench
```

### **Quick Test:**
```bash
# Test with sample data
python -m src.official_approach `
  --model_path outputs/research_model/latest_model `
  --dev_csv trial.csv `
  --output_json test_submission.json
```

## **📞 Support and Resources**

### **Official Resources:**
- **Starter Kit**: https://noshinulfat.github.io/blp25_code_generation_task/
- **CodaBench**: https://codalab.lisn.upsaclay.fr/competitions/
- **Paper Template**: ACL 2023 style, 4 pages

### **Research Model Features:**
- **Dual-Stream Architecture**: Parallel instruction and code processing
- **Code-Aware Attention**: Syntax-aware attention mechanism
- **Bangla-Specific**: Optimized for Bangla language patterns
- **Efficient Training**: Faster convergence with fewer parameters

---

**🎯 This approach combines the best of both worlds: our novel research architecture with official task compatibility, aiming for state-of-the-art performance on the BLP25 challenge.**
