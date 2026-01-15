# Testing Commands for sentence_level_error.py

This file contains various test commands to verify `sentence_level_error.py` is working correctly.

## Quick Tests (Small Samples)

### Test 1: Basic MT Analysis (Swahili, 10 samples)
```bash
python sentence_level_error.py --task mt --config health --lang sw --max-samples 10
```
**Expected**: Should complete in ~2-3 minutes and create files in `quality_reports/`

### Test 2: Different Language (Yoruba)
```bash
python sentence_level_error.py --task mt --config health --lang yo --max-samples 10
```

### Test 3: Tech Domain
```bash
python sentence_level_error.py --task mt --config tech --lang sw --max-samples 10
```

### Test 4: NER Example
```bash
python sentence_level_error.py --task ner
```
**Expected**: Should create `ner_quality_report.txt` with annotation consistency errors

## Medium Tests (100 samples)

### Test 5: Standard Analysis
```bash
python sentence_level_error.py --task mt --config health --lang sw --max-samples 100
```
**Expected**: ~10-15 minutes on GPU, creates detailed reports

### Test 6: All Languages (Health Domain)
```bash
for lang in am ha sw yo zu; do
    python sentence_level_error.py --task mt --config health --lang $lang --max-samples 50
done
```
**Windows PowerShell:**
```powershell
foreach ($lang in @('am','ha','sw','yo','zu')) {
    python sentence_level_error.py --task mt --config health --lang $lang --max-samples 50
}
```

## Batch Tests

### Test 7: Using Batch Runner
```bash
# First, edit batch_analyze.py to set MAX_SAMPLES = 20
python batch_analyze.py
```
**Expected**: Analyzes all config/language combinations, saves to `batch_analysis_results/`

## Comparison Tests

### Test 8: Compare Languages
```bash
# After running Test 6 or 7
python compare_results.py --compare languages --config health
```

### Test 9: Compare Configs
```bash
# Run analysis for both configs
python sentence_level_error.py --task mt --config health --lang sw --max-samples 50
python sentence_level_error.py --task mt --config tech --lang sw --max-samples 50

# Compare
python compare_results.py --compare configs --lang sw
```

## Programmatic Tests

### Test 10: Python API
```python
from sentence_level_error import load_afridocmt_for_mt, DataQualityChecker, TaskType

# Load data
datasets = load_afridocmt_for_mt(config_name="health", max_samples=5)
print(f"Loaded {len(datasets['sw'])} Swahili examples")

# Analyze
checker = DataQualityChecker("facebook/nllb-200-distilled-600M", TaskType.MT)
errors = checker.check_all_errors(datasets['sw'])
print(f"Found {len(errors)} errors")

# Report
checker.generate_report(errors, "test_report.txt")
```

## Verification Checklist

After running tests, verify:

- [ ] Files created in `quality_reports/`:
  - [ ] `quality_report_*.txt` (text report)
  - [ ] `errors_*.json` (JSON export)

- [ ] Text report contains:
  - [ ] Summary statistics
  - [ ] Error breakdown by type
  - [ ] Top 5 examples per error type
  - [ ] Evidence and recommendations

- [ ] JSON export contains:
  - [ ] List of error objects
  - [ ] Each error has: type, confidence, details, recommendations

- [ ] Console output shows:
  - [ ] Progress messages (loading dataset, initializing, detecting)
  - [ ] Success confirmation
  - [ ] File paths to generated reports

## Expected Error Types for MT

You should see some of these error types in the reports:

1. **SEMANTIC_DRIFT** - Translation doesn't match source meaning
2. **TRANSLATIONESE** - Word-for-word translation (unnatural)
3. **CODE_SWITCHING_INCONSISTENCY** - Inconsistent term handling

Note: Not all error types will appear in every run. It depends on the data quality.

## Troubleshooting Test Results

### No errors detected
- This is possible if the dataset quality is good
- Try increasing `--max-samples` to find edge cases
- Check the report for "Clean dataset!" message

### Out of memory
- Reduce `--max-samples` to 10 or 20
- Use smaller model: `--model facebook/m2m100_418M`
- Close other GPU-using applications

### Dataset download fails
- Check internet connection
- Try running twice (first run downloads, second processes)
- Manually download: `python -c "from datasets import load_dataset; load_dataset('masakhane/AfriDocMT', 'health')"`

### Model download fails
- Check HuggingFace is accessible
- Try downloading manually first in Python:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
```

## Performance Benchmarks

Expected runtimes (approximate, on GPU):

| Samples | Language | Expected Time |
|---------|----------|---------------|
| 10      | Any      | 1-3 min       |
| 50      | Any      | 5-10 min      |
| 100     | Any      | 10-20 min     |
| 500     | Any      | 40-90 min     |

CPU times are typically 3-5x longer.

## Test Output Examples

### Good Output
```
================================================================================
MACHINE TRANSLATION QUALITY ANALYSIS
================================================================================
Model: facebook/nllb-200-distilled-600M
Dataset: AfriDocMT/health
Target Language: sw
Max Samples: 10
================================================================================

[1/3] Loading dataset...
✓ Loaded 10 translation pairs

[2/3] Initializing quality checker...
✓ Initialized machine_translation checker with facebook/nllb-200-distilled-600M

[3/3] Detecting errors...
Checking semantic drift...
Checking translationese...
Checking code-switching inconsistency...

✓ Report generated: quality_reports/quality_report_health_sw.txt
  Total errors: 3
  Error types: 2

✓ Errors exported to JSON: quality_reports/errors_health_sw.json

================================================================================
ANALYSIS COMPLETE
================================================================================
```

### Error Output
```
ERROR: Failed to load dataset: Connection error...
INFO: Attempting to download from HuggingFace...
```

This is normal on first run - the dataset is being downloaded.

## Next Steps After Testing

Once tests pass:

1. Scale up to larger samples (100-500)
2. Run batch analysis for all languages
3. Compare results across languages/configs
4. Analyze the generated reports for insights
5. Use findings to improve dataset quality
