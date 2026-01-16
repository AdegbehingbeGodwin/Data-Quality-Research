# Data Quality Research - Command Cheat Sheet

## Quick Commands

### Installation
```bash
# Windows
setup.bat

# Linux/Mac  
bash setup.sh

# Verify
python test_setup.py
```

### sentence_level_error.py

```bash
# Quick test (10 samples)
python sentence_level_error.py --task mt --config health --lang sw --max-samples 10

# Standard (100 samples)
python sentence_level_error.py --task mt --config health --lang sw --max-samples 100

# All languages
for lang in am ha sw yo zu; do python sentence_level_error.py --task mt --config health --lang $lang --max-samples 50; done

# NER example
python sentence_level_error.py --task ner
```

### run_experiments.py

```bash
# Single experiment
python run_experiments.py --config health --lang sw

# Sentence-level only
python run_experiments.py --sentence-only

# Full suite
python run_experiments.py --all

# Custom
python run_experiments.py --config health --lang sw --max-samples 1000 --output-dir my_results
```

### Batch & Compare

```bash
# Batch analyze all
python batch_analyze.py

# Compare languages
python compare_results.py --compare languages --config health

# Compare configs
python compare_results.py --compare configs --lang sw

# Compare both
python compare_results.py --compare both
```

## Output Locations

```
sentence_level_error.py → quality_reports/
run_experiments.py      → results_complete/
batch_analyze.py        → batch_analysis_results/
```

## Common Options

| Option | Values | Default |
|--------|--------|---------|
| --config | health, tech, doc_* | health |
| --lang | am, ha, sw, yo, zu | sw |
| --max-samples | any number | 100/500 |
| --task | mt, ner, qa | mt |
| --model | HF model name | nllb-200-distilled-600M |

## File Structure

```
Data Quality Research/
├── sentence_level_error.py   # Multi-task framework
├── run_experiments.py         # MT experiments
├── batch_analyze.py           # Batch processing
├── compare_results.py         # Result comparison
├── test_setup.py              # Verification
├── config.py                  # Configuration
├── utils.py                   # Utilities
├── error_detector.py          # Detection logic
├── requirements.txt           # Dependencies
└── README.md                  # Full docs
```

## Quick Troubleshooting

```bash
# Out of memory
--max-samples 10

# Slow? Use smaller model
--model "facebook/m2m100_418M"

# Check dependencies
pip install -r requirements.txt

# Test GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Documentation

- `README.md` - Complete guide
- `QUICKSTART.md` - Quick reference  
- `SENTENCE_LEVEL_USAGE.md` - Detailed usage
- `TESTING.md` - Test commands
- `PROJECT_STATUS.md` - What's ready
- `IMPROVEMENTS.md` - What changed

## Languages

| Code | Language | NLLB Code |
|------|----------|-----------|
| am   | Amharic  | amh_Ethi  |
| ha   | Hausa    | hau_Latn  |
| sw   | Swahili  | swh_Latn  |
| yo   | Yoruba   | yor_Latn  |
| zu   | Zulu     | zul_Latn  |
