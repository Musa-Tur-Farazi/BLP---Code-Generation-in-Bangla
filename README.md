# 🎯 BLP25 Bangla-to-Code Generation

**Official Task**: Generate Python functions from Bangla instructions  
**Evaluation**: Pass@1 (percentage of test cases that pass)  
**Deadline**: September 29, 2025

## **📁 Project Structure**

```
├── README.md                    # This file
├── OFFICIAL_TASK_GUIDE.md      # Complete guide with commands
├── requirements.txt             # Python dependencies
├── trial.csv                   # Training data (with solutions)
├── dev_v2.csv                  # Development data (no solutions)
├── data/
│   └── train.json              # Converted training data
├── src/
│   ├── research_model.py       # Custom dual-stream architecture
│   ├── research_training.py    # Training script
│   ├── official_approach.py    # Generate submission
│   ├── convert_csv_to_json.py  # Data conversion
│   ├── data_utils.py           # Data utilities
│   └── prompting.py            # Prompt templates
└── outputs/                    # Trained models
```

## **🚀 Quick Start**

### **1. Setup Environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### **2. Prepare Data**
```bash
python -m src.convert_csv_to_json --input_csv trial.csv --output_json data/train.json
```

### **3. Train Model**
```bash
python -m src.research_training `
  --train_json data/train.json `
  --model_size small `
  --use_dual_stream `
  --num_epochs 10 `
  --output_dir outputs/research_model
```

### **4. Generate Submission**
```bash
python -m src.official_approach `
  --model_path outputs/research_model/latest_model `
  --dev_csv dev_v2.csv `
  --create_zip
```

### **5. Submit to CodaBench**
Upload `submission.zip` to the dev phase.

## **🔬 Research Model**

**Novel Dual-Stream Architecture:**
- **Instruction Stream**: Bangla language processing
- **Code Stream**: Python function generation
- **Cross-Attention**: Bidirectional information flow
- **Code-Aware Attention**: Syntax-aware mechanisms

**Expected Performance:** 60-80% Pass@1 (vs 10-20% baseline)

## **📊 Model Sizes**

| Size | Parameters | Training Time | Use Case |
|------|------------|---------------|----------|
| Tiny | ~2M | 5 min | Quick test |
| Small | ~15M | 15 min | Development |
| Medium | ~50M | 30 min | Production |

## **📝 Submission Format**

```json
[
  {
    "id": 1,
    "response": "```python\ndef function_name(...):\n    # implementation\n    return result\n```"
  }
]
```

## **🔗 Resources**

- **Official Task**: https://noshinulfat.github.io/blp25_code_generation_task/
- **CodaBench**: https://codalab.lisn.upsaclay.fr/competitions/
- **Complete Guide**: See `OFFICIAL_TASK_GUIDE.md`

---

**🎯 This project implements a novel dual-stream architecture for Bangla-to-Python code generation, combining research innovation with official task compatibility.**
