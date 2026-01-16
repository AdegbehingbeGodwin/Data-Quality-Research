# Quick Reference Guide

## Installation
```bash
# Windows
setup.bat

# Linux/Mac
bash setup.sh

# Manual
pip install -r requirements.txt
python test_setup.py
```

## Running Experiments

### Option 1: Sentence-level only (uses local CSV data)
```bash
python run_experiments.py --sentence-only
```

### Option 2: Single experiment
```bash
python run_experiments.py --config health --lang sw
```

### Option 3: Full suite (all configs and languages)
```bash
python run_experiments.py --all
```

### Option 4: Custom settings
```bash
python run_experiments.py \
    --config doc_health_10 \
    --lang yo \
    --max-samples 1000 \
    --output-dir my_results
```

## Configuration

### Edit thresholds in `config.py`:
```python
omission_threshold = 0.7           # Hypothesis < 70% of reference
low_confidence_threshold = -6.0    # Log prob threshold
high_conf_error_similarity = 0.5   # Similarity threshold
```

### Change model:
```python
model_name = "facebook/nllb-200-distilled-600M"  # In config.py
```

## Programmatic Usage

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

## Output Location

Results saved to `results_complete/` (or custom `--output-dir`):
```
results_complete/
├── master_summary.json          # Overall stats
├── experiment.log               # Debug logs
└── [config]/[lang]/
    ├── case_studies.json        # Detailed errors
    └── summary.json             # Error counts
```

## GPU Settings

Auto-detected by default. To force CPU:
```python
# In config.py, change:
device = "cpu"
```

Multi-GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --all
```

## Supported Languages

| Code | Language | NLLB Code |
|------|----------|-----------|
| `am` | Amharic  | amh_Ethi  |
| `ha` | Hausa    | hau_Latn  |
| `sw` | Swahili  | swh_Latn  |
| `yo` | Yoruba   | yor_Latn  |
| `zu` | Zulu     | zul_Latn  |

## Available Configs

**Sentence-level** (local CSV data):
- `tech`
- `health`

**Document-level** (requires HuggingFace download):
- `doc_tech`, `doc_health`
- `doc_tech_5`, `doc_health_5`
- `doc_tech_10`, `doc_health_10`
- `doc_tech_25`, `doc_health_25`

## Error Types

1. **OMISSION**: Translation too short
2. **REPETITION**: Duplicate tokens
3. **TERMINOLOGY_DRIFT**: Inconsistent terms
4. **LOW_CONFIDENCE**: Model uncertain
5. **HIGH_CONFIDENCE_ERROR**: Confident but wrong

## Troubleshooting

### Out of GPU memory
```bash
# Reduce max_length in config.py
max_length = 256  # Default: 512
```

### Dataset not found
```bash
# Download manually
python -c "from datasets import load_dataset; load_dataset('masakhane/AfriDocMT', 'health')"
```

### Import errors
```bash
pip install -r requirements.txt --upgrade
```

## Estimation

### Time per experiment:
- Sentence-level (500 samples): ~15-30 min
- Document-level (500 samples): ~20-40 min

### Full suite (50 combinations):
- GPU: 12-25 hours
- CPU: 50-100 hours

## Key Files

- `run_experiments.py` - Main runner
- `config.py` - All settings
- `error_detector.py` - Detection logic
- `utils.py` - Helper functions
- `test_setup.py` - Verification
- `README.md` - Full documentation
