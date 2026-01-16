# Data Quality Research for African Language MT

Automated detection of data quality errors in African language machine translation using model explainability techniques.

## 📖 Overview

This project analyzes the **AfriDocMT dataset** (Masakhane's African Document-level MT corpus) to identify various types of translation errors using:
- Attention mechanism analysis
- Gradient-based attribution
- Heuristic error detection

**Supported Languages**: Amharic, Hausa, Swahili, Yoruba, Zulu  
**Domains**: Health, Technology  
**Model**: NLLB-200-distilled-600M

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

**Option 1: Single config/language pair**
```bash
python run_experiments.py --config health --lang sw
```

**Option 2: Sentence-level only (uses local CSV data)**
```bash
python run_experiments.py --sentence-only
```

**Option 3: Full experiment suite** (requires HuggingFace dataset download)
```bash
python run_experiments.py --all
```

### Custom Settings

```bash
python run_experiments.py \
    --config doc_health_10 \
    --lang yo \
    --max-samples 1000 \
    --output-dir my_results
```

## 📁 Project Structure

```
Data Quality Research/
├── config.py                    # Configuration management
├── utils.py                     # Utility functions
├── error_detector.py            # Error detection logic
├── run_experiments.py           # Main experiment runner
├── sentence_level_error.py      # Original framework (multi-task)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── Health/                      # Healthcare domain sentence data
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
│
└── Tech/                        # Technology domain sentence data
    ├── train.csv
    ├── dev.csv
    └── test.csv
```

## 🔍 Error Types Detected

1. **OMISSION**: Translation significantly shorter than reference
2. **REPETITION**: Duplicate tokens in output
3. **TERMINOLOGY_DRIFT**: Inconsistent term translation within document
4. **LOW_CONFIDENCE**: Model uncertainty (low log probability)
5. **HIGH_CONFIDENCE_ERROR**: High confidence but poor quality

## ⚙️ Configuration

Edit `config.py` to customize:
- Model selection and max length
- Error detection thresholds
- Output settings
- Dataset configurations

### Key Thresholds

```python
omission_threshold = 0.7           # <70% length triggers omission
low_confidence_threshold = -6.0    # Log prob threshold
high_conf_error_similarity = 0.5   # Similarity threshold
terminology_drift_threshold = 0.2  # 20% inconsistency
```

## 📊 Output Format

Results are saved to `results_complete/` (or custom dir):

```
results_complete/
├── master_summary.json          # Overall statistics
├── experiment.log               # Detailed logs
│
└── [config_name]/
    └── [language]/
        ├── case_studies.json    # Detailed error cases with explainability
        └── summary.json         # Error counts
```

### Case Study Format

Each error case includes:
- Source and reference texts
- Generated hypothesis
- Error types detected
- Token-level gradient attribution
- Cross-attention weights
- Confidence scores

## 🧪 GPU Acceleration

The code automatically uses GPU if available:

```python
# In config.py
device = "cuda"  # Auto-falls back to "cpu" if unavailable
```

For multi-GPU systems, set:
```bash
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --all
```

## 📝 Dataset Information

**Source**: [masakhane/AfriDocMT](https://huggingface.co/datasets/masakhane/AfriDocMT)

**Available Configurations**:
- **Sentence-level**: `tech`, `health` (available locally as CSVs)
- **Document-level**: `doc_tech`, `doc_health`, `doc_tech_5/10/25`, `doc_health_5/10/25`

### Downloading Full Dataset

Document-level configs require HuggingFace download:

```python
from datasets import load_dataset

# Download specific config
dataset = load_dataset("masakhane/AfriDocMT", "doc_health_10")

# Or in CLI (automatic on first run)
python run_experiments.py --config doc_health_10 --lang sw
```

## 🔬 Advanced Usage

### Custom Error Detector

```python
from error_detector import MTErrorDetector

detector = MTErrorDetector(
    omission_threshold=0.6,        # More sensitive
    low_conf_threshold=-5.0,       # Stricter
    terminology_drift_threshold=0.15
)
```

### Programmatic Access

```python
from run_experiments import AfriDocMTExperiment
from config import experiment_config

# Customize config
experiment_config.max_samples = 1000
experiment_config.verbose = True

# Run experiment
exp = AfriDocMTExperiment()
summary = exp.run_single_config("health", "sw")
print(summary)
```

## 📚 Reference

Based on research from:
- **Paper**: "AFRIDOC-MT: Document-level MT Corpus for African Languages" (2501.06374v2.pdf)
- **Dataset**: [Masakhane AfriDocMT](https://huggingface.co/datasets/masakhane/AfriDocMT)
- **Model**: [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)

## 🛠️ Troubleshooting

### Out of Memory (GPU)
- Reduce `max_samples` or `max_length` in config
- Use smaller batch size (currently 1)
- Fall back to CPU: `device = "cpu"` in config

### Dataset Not Found
```bash
# Download specific config
python -c "from datasets import load_dataset; load_dataset('masakhane/AfriDocMT', 'health')"
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📄 License

See project license file.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional error detection heuristics
- Support for more languages
- Visualization tools for attention/gradients
- Statistical significance testing
- Human validation interface

## 📧 Contact

For questions about the research or code, please open an issue.
