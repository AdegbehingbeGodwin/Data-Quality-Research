# Usage Guide for sentence_level_error.py

## Overview

The `sentence_level_error.py` script is a comprehensive framework for detecting data quality errors in African language NLP datasets. It now includes built-in support for AfriDocMT dataset from HuggingFace.

## Quick Start

### 1. Basic MT Analysis (Machine Translation)

```bash
# Analyze Swahili translations from Health domain
python sentence_level_error.py --task mt --config health --lang sw --max-samples 100

# Analyze Yoruba translations from Tech domain
python sentence_level_error.py --task mt --config tech --lang yo --max-samples 200

# Analyze all languages (will take longer)
for lang in am ha sw yo zu; do
    python sentence_level_error.py --task mt --config health --lang $lang --max-samples 50
done
```

### 2. Custom Model

```bash
# Use a different MT model
python sentence_level_error.py \
    --task mt \
    --config health \
    --lang sw \
    --model "facebook/m2m100_418M" \
    --max-samples 100
```

### 3. NER Analysis Example

```bash
# Run the built-in NER example
python sentence_level_error.py --task ner
```

## Output Files

Results are saved in `quality_reports/` directory:

```
quality_reports/
├── quality_report_health_sw.txt    # Human-readable report
└── errors_health_sw.json           # Machine-readable errors
```

### Report Contents

**Text Report (`quality_report_*.txt`):**
- Summary statistics (total errors, error types)
- Average confidence per error type
- Top 5 examples per error type with evidence

**JSON Export (`errors_*.json`):**
- Structured error data
- Machine-readable format
- Includes error type, confidence, details, recommendations

## Available Languages

All African languages from AfriDocMT:

| Code | Language | Script      |
|------|----------|-------------|
| `am` | Amharic  | Ethiopic    |
| `ha` | Hausa    | Latin       |
| `sw` | Swahili  | Latin       |
| `yo` | Yoruba   | Latin       |
| `zu` | Zulu     | Latin       |

## Dataset Configurations

### Sentence-level (Available)
- `health` - Healthcare domain
- `tech` - Technology domain

### Document-level (Will be downloaded from HuggingFace)
- `doc_health`, `doc_tech`
- `doc_health_5`, `doc_tech_5` (5 sentences per document)
- `doc_health_10`, `doc_tech_10` (10 sentences)
- `doc_health_25`, `doc_tech_25` (25 sentences)

## Programmatic Usage

### Example 1: Custom Analysis

```python
from sentence_level_error import (
    DataQualityChecker, 
    TaskType, 
    load_afridocmt_for_mt
)

# Load dataset
mt_datasets = load_afridocmt_for_mt(config_name="health", max_samples=50)
swahili_data = mt_datasets["sw"]

# Initialize checker
checker = DataQualityChecker("facebook/nllb-200-distilled-600M", TaskType.MT)

# Run analysis
errors = checker.check_all_errors(swahili_data)

# Generate report
checker.generate_report(errors, "my_custom_report.txt")
```

### Example 2: Batch Processing

```python
import os
from sentence_level_error import run_mt_quality_analysis

languages = ["sw", "yo", "ha"]
configs = ["health", "tech"]

for config in configs:
    for lang in languages:
        print(f"\nAnalyzing {config}/{lang}...")
        run_mt_quality_analysis(
            config_name=config,
            target_lang=lang,
            max_samples=100,
            output_dir=f"batch_results/{config}"
        )
```

### Example 3: Custom Error Detection

```python
from sentence_level_error import DataQualityChecker, TaskType

# Initialize with custom thresholds (modify the class internally)
checker = DataQualityChecker("facebook/nllb-200-distilled-600M", TaskType.MT)

# Prepare your custom dataset
my_dataset = [
    {
        "source": "The patient has a fever.",
        "target": "Mgonjwa ana homa.",
        "id": "001"
    },
    # ... more examples
]

# Run detection
errors = checker.check_all_errors(my_dataset)

# Analyze results
for error in errors:
    print(f"Error: {error.error_type}")
    print(f"Confidence: {error.confidence:.3f}")
    print(f"Details: {error.details}\n")
```

## Error Types Detected

### Machine Translation Errors

1. **SEMANTIC_DRIFT**
   - Detection: Cross-attention entropy > 0.8
   - Meaning: Source and target are not semantically aligned
   - Action: Verify translations convey same meaning

2. **TRANSLATIONESE**
   - Detection: Diagonal attention ratio > 0.4
   - Meaning: Word-for-word translation (unnatural target)
   - Action: Check if target sounds natural

3. **CODE_SWITCHING_INCONSISTENCY**
   - Detection: Same word copied vs. translated > 20% inconsistency
   - Meaning: Inconsistent handling of code-switched terms
   - Action: Establish consistent guidelines

### Token-Level Errors (NER/POS)

1. **ANNOTATION_INCONSISTENCY**
   - Same token with different labels across dataset

2. **LABEL_NOISE**
   - Model strongly predicts different label than annotation

3. **SPURIOUS_CORRELATION**
   - Labels correlated with position rather than content

### Other Task Errors

- **QA_MISALIGNMENT**: Question-answer type mismatch
- **CONTEXT_CONTRADICTION**: Answer contradicts context
- **SENTIMENT_LABEL_NOISE**: Possible sarcasm or mislabeling

## Performance Considerations

### Memory Usage

**Model Loading:**
- NLLB-200-distilled-600M: ~2.5 GB GPU memory
- NLLB-200-1.3B: ~5 GB GPU memory
- M2M100-418M: ~1.7 GB GPU memory

**Dataset Processing:**
- 100 samples: ~5-10 minutes
- 500 samples: ~20-40 minutes
- 1000 samples: ~40-80 minutes

### Optimization Tips

```python
# 1. Reduce max_length for faster processing
# Edit in the code or use smaller model

# 2. Process in batches
for i in range(0, 1000, 100):
    mt_datasets = load_afridocmt_for_mt(
        config_name="health", 
        max_samples=100
    )
    # Process batch...

# 3. Use CPU if GPU memory limited
# The script auto-detects, or force CPU by setting device in code
```

## Comparison with run_experiments.py

| Feature | sentence_level_error.py | run_experiments.py |
|---------|------------------------|-------------------|
| **Focus** | General framework | AfriDocMT specific |
| **Tasks** | MT, NER, POS, QA, Sentiment | MT only |
| **Config** | Hardcoded in script | Centralized config.py |
| **Explainability** | Full (attention + gradients) | Full (attention + gradients) |
| **Output** | Text report + JSON | JSON case studies |
| **CLI** | Basic | Advanced with many options |
| **Best For** | Exploratory analysis | Production experiments |

## When to Use Which

**Use `sentence_level_error.py` when:**
- You want to quickly test on a small sample
- You're exploring different task types (NER, QA, etc.)
- You need a simple text report
- You're prototyping new error detection methods

**Use `run_experiments.py` when:**
- Running large-scale experiments
- Need consistent configuration management
- Want structured JSON output
- Need logging and progress tracking
- Working with multiple configs/languages systematically

## Troubleshooting

### Issue: Out of Memory

```bash
# Solution 1: Use smaller model
python sentence_level_error.py --model "facebook/m2m100_418M" --task mt --config health --lang sw

# Solution 2: Reduce samples
python sentence_level_error.py --max-samples 50 --task mt --config health --lang sw

# Solution 3: Force CPU (edit script to set device="cpu")
```

### Issue: Dataset Download Fails

```python
# Download manually first
from datasets import load_dataset
dataset = load_dataset("masakhane/AfriDocMT", "health")

# Then run the script
python sentence_level_error.py --task mt --config health --lang sw
```

### Issue: Model Download Fails

```python
# Pre-download model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Then run the script
```

### Issue: Import Errors

```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Check specific package
python -c "import transformers; print(transformers.__version__)"
```

## Advanced: Extending the Framework

### Adding a New Error Type

```python
# 1. Add detection method to DataQualityChecker class
def _detect_my_custom_error(self, dataset: List[Dict]) -> List[ErrorReport]:
    errors = []
    # Your detection logic here
    return errors

# 2. Add to check_all_errors method
def check_all_errors(self, dataset: List[Dict]) -> List[ErrorReport]:
    all_errors = []
    # ... existing checks
    all_errors.extend(self._detect_my_custom_error(dataset))
    return all_errors
```

### Adding a New Task Type

```python
# 1. Add to TaskType enum
class TaskType(Enum):
    # ... existing tasks
    SUMMARIZATION = "summarization"

# 2. Add task-specific checker method
def _check_summarization_errors(self, dataset: List[Dict]) -> List[ErrorReport]:
    errors = []
    # Your detection logic
    return errors

# 3. Add model loading logic
def __init__(self, model_name: str, task_type: TaskType, device: str = "cuda"):
    # ... existing code
    elif task_type == TaskType.SUMMARIZATION:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

## Examples Gallery

### Example: Analyze All Languages

```bash
#!/bin/bash
# analyze_all_languages.sh

for lang in am ha sw yo zu; do
    echo "Processing $lang..."
    python sentence_level_error.py \
        --task mt \
        --config health \
        --lang $lang \
        --max-samples 100 \
        --output-dir "results_by_language"
done

echo "Analysis complete! Check results_by_language/"
```

### Example: Compare Health vs Tech

```python
from sentence_level_error import run_mt_quality_analysis

# Analyze both domains for Swahili
health_errors = run_mt_quality_analysis(
    config_name="health",
    target_lang="sw",
    max_samples=100,
    output_dir="comparison"
)

tech_errors = run_mt_quality_analysis(
    config_name="tech",
    target_lang="sw",
    max_samples=100,
    output_dir="comparison"
)

print(f"Health errors: {len(health_errors)}")
print(f"Tech errors: {len(tech_errors)}")
```

## Summary

The `sentence_level_error.py` script is now **fully functional** and ready to use with:

✅ Built-in AfriDocMT dataset loading  
✅ Automatic HuggingFace download  
✅ CLI interface for easy usage  
✅ Programmatic API for custom workflows  
✅ Comprehensive error detection  
✅ Human-readable and JSON reports  
✅ Multi-task support (MT, NER, QA, etc.)  

**Next Steps:**
1. Run a small test: `python sentence_level_error.py --task mt --config health --lang sw --max-samples 10`
2. Review the generated reports in `quality_reports/`
3. Scale up to larger samples
4. Analyze multiple languages/domains
