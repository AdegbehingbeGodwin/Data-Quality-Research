# Data Quality Research - Project Navigation Guide

**Last Updated:** January 16, 2026  
**Status:** Organized & Clean  
**Project:** Comprehensive Data Quality Analysis for African Language Machine Translation

---

## 📁 Directory Structure

```
data-quality-research/
├── docs/                          # 📚 All documentation and reports
├── scripts/                       # 🐍 Python scripts and utilities
├── data_analysis/results/         # 📊 Analysis results, visualizations, data
├── setup/                         # ⚙️ Installation and setup scripts
├── Health/                        # 🏥 Health domain dataset (CSV files)
├── Tech/                          # 💻 Technology domain dataset (CSV files)
├── temp_files/                    # 🗑️ Temporary files and logs
├── requirements.txt               # 📦 Python dependencies
├── .gitignore                     # 🔒 Git ignore rules
└── .venv/                         # 🔧 Virtual environment
```

---

## 📚 Documentation (`docs/`)

**Start Here:**

### Getting Started
- **[README.md](docs/README.md)** - Project overview and quick start
- **[QUICKSTART.md](docs/QUICKSTART.md)** - Quick start guide (if different)
- **[QUICK_START_GUIDE.md](docs/QUICK_START_GUIDE.md)** - Extended quick start

### Research & Methodology
- **[METHODOLOGY_AND_DISCUSSION.md](docs/METHODOLOGY_AND_DISCUSSION.md)** ⭐ **MAIN REFERENCE**
  - Complete methodology documentation
  - Error classification framework (4 categories, 12 types)
  - Implementation details
  - Results analysis
  - Discussion of findings
  - Recommendations

### Analysis Reports
- **[DETAILED_QUALITY_ANALYSIS_REPORT.md](docs/DETAILED_QUALITY_ANALYSIS_REPORT.md)** - Comprehensive findings
- **[DATA_QUALITY_FINDINGS.md](docs/DATA_QUALITY_FINDINGS.md)** - Summary of data quality issues
- **[EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md)** - Executive summary
- **[README_ANALYSIS_RESULTS.md](docs/README_ANALYSIS_RESULTS.md)** - Analysis results documentation

### Visualizations Guide
- **[VISUALIZATION_REPORT.md](docs/VISUALIZATION_REPORT.md)** ⭐ **VISUALIZATIONS**
  - All 7 visualizations documented
  - Validation report
  - Data quality metrics per visualization
- **[VISUALIZATIONS_GUIDE.md](docs/VISUALIZATIONS_GUIDE.md)** - Guide to interpreting visualizations

### Usage & Examples
- **[SENTENCE_LEVEL_USAGE.md](docs/SENTENCE_LEVEL_USAGE.md)** - How to use sentence-level error detection
- **[CHEATSHEET.md](docs/CHEATSHEET.md)** - Quick reference and cheatsheet
- **[dataset.md](docs/dataset.md)** - Dataset documentation

### Project Management
- **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Project status and tracking
- **[TESTING.md](docs/TESTING.md)** - Testing documentation
- **[EXPERIMENT_REVIEW.md](docs/EXPERIMENT_REVIEW.md)** - Experiment review
- **[IMPROVEMENTS.md](docs/IMPROVEMENTS.md)** - Suggested improvements

### Additional Resources
- **[2501.06374v2.pdf](docs/2501.06374v2.pdf)** - Reference paper
- **[data quality research document.docx](docs/data%20quality%20research%20document.docx)** - Word document

---

## 🐍 Scripts (`scripts/`)

### Main Analysis Scripts
- **comprehensive_quality_analysis_enhanced.py** - ⭐ **RUN THIS** 
  - Main comprehensive analysis script
  - Combines all data splits (train+dev+test)
  - Generates visualizations and reports
  - **Usage:** `python scripts/comprehensive_quality_analysis_enhanced.py --output data_analysis/results`

- **comprehensive_quality_analysis.py** - Original version (test-only)
- **batch_analyze.py** - Batch analysis script

### Error Detection
- **enhanced_error_detection.py** - ⭐ **Core error detection engine**
  - DetailedQualityAnalyzer class
  - 12 error detection methods
  - 4 error categories
  - **Key Class:** `DetailedQualityAnalyzer`

- **sentence_level_error.py** - Model-based error detection
  - DataQualityChecker class
  - Semantic drift, translationese, code-switching
  - **Key Class:** `DataQualityChecker`

- **error_detector.py** - Original error detector

### Utilities & Configuration
- **config.py** - Configuration settings
- **utils.py** - Utility functions
- **compare_results.py** - Compare analysis results
- **run_experiments.py** - Run experiments
- **test_setup.py** - Test setup validation

### Setup & Installation
Located in `setup/` folder:
- **setup.sh** - Linux/Mac setup
- **setup.bat** - Windows setup
- Run before first use

---

## 📊 Data Analysis Results (`data_analysis/results/`)

### Data Files (JSON format)
- **health_am_results_enhanced.json** - Amharic + Health domain
- **health_ha_results_enhanced.json** - Hausa + Health domain ⚠️ **CRITICAL FINDINGS**
- **health_sw_results_enhanced.json** - Swahili + Health domain
- **health_yo_results_enhanced.json** - Yoruba + Health domain
- **health_zu_results_enhanced.json** - Zulu + Health domain
- **tech_am_results_enhanced.json** - Amharic + Tech domain
- **tech_ha_results_enhanced.json** - Hausa + Tech domain ⚠️ **CRITICAL FINDINGS**
- **tech_sw_results_enhanced.json** - Swahili + Tech domain
- **tech_yo_results_enhanced.json** - Yoruba + Tech domain
- **tech_zu_results_enhanced.json** - Zulu + Tech domain

### Summary Files
- **summary_enhanced.json** - Complete summary of all results
- **comprehensive_analysis_enhanced.txt** - Text report of analysis

### Visualizations (`data_analysis/results/visualizations/`)
**All PNG images at 300 DPI resolution:**

1. **06_split_comparison.png** - Error rates by data split (train/dev/test)
2. **07_error_types_detailed.png** - Error types per language
3. **08_sample_coverage.png** - Sample distribution coverage
4. **09_split_error_tables.png** - Detailed error tables by split
5. **10_error_type_breakdown.png** ⭐ **Ranked error types**
6. **11_quality_issues_heatmap.png** ⭐ **Overview heatmap**
7. **12_detailed_quality_issues.png** ⭐ **Main visualization** (4 error categories)

---

## 📁 Datasets (`Health/` & `Tech/`)

### Available Splits
Each domain has three CSV files:
- **train.csv** - Training data (~7,000 samples)
- **dev.csv** - Development data (~970 samples)
- **test.csv** - Test data (~1,982 samples)

### Languages
- **am** - Amharic
- **ha** - Hausa
- **sw** - Swahili
- **yo** - Yoruba
- **zu** - Zulu

### Usage
```python
import pandas as pd
# Load Amharic health domain test data
df = pd.read_csv("Health/test.csv")
# Access translations: df['en'] (source), df['am'] (Amharic)
```

---

## 📦 Requirements & Setup

### Installation
```bash
# Navigate to project directory
cd data-quality-research

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- pandas, numpy - Data processing
- matplotlib, seaborn - Visualization
- torch - Deep learning
- transformers - Hugging Face models
- captum - Attribution analysis

---

## 🚀 Quick Start

### Run Full Analysis
```bash
cd data-quality-research
python scripts/comprehensive_quality_analysis_enhanced.py --output data_analysis/results
```

### Check Results
```bash
# View visualizations
ls -la data_analysis/results/visualizations/

# Check summary
cat data_analysis/results/summary_enhanced.json | python -m json.tool

# Read main report
cat data_analysis/results/comprehensive_analysis_enhanced.txt
```

### Use Error Detection in Code
```python
from scripts.enhanced_error_detection import DetailedQualityAnalyzer

# Create analyzer
analyzer = DetailedQualityAnalyzer()

# Analyze dataset
results = analyzer.analyze_all_issues(your_dataset)
print(results)
```

---

## 🔍 Key Findings Summary

**Analysis Scope:** 100,000 samples (10,000 per language/domain)

### Critical Issues Found ⚠️
- **Hausa Off-target Language:** 9,998 instances (99.3% of test samples)
- **Amharic Script Mixing:** 794 instances (Ge'ez + Latin)
- **Lexical Cohesion Errors:** 6,265 instances across all languages

### Error Distribution
- Generation Errors: 11,108 (62.2%)
- Cohesion Errors: 6,313 (35.4%)
- Linguistic Issues: 878 (4.9%)
- Data Noise: 151 (0.8%)

### Split Analysis
- Train errors: 0% (clean)
- Dev errors: 0% (clean)
- Test errors: 48.11% (problematic)

---

## 📖 Reading Guide by Role

### For Project Managers
1. Read: [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)
2. Review: [EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md)
3. Check: [11_quality_issues_heatmap.png](data_analysis/results/visualizations/11_quality_issues_heatmap.png)

### For Researchers
1. Start: [METHODOLOGY_AND_DISCUSSION.md](docs/METHODOLOGY_AND_DISCUSSION.md) ⭐
2. Deep dive: [DETAILED_QUALITY_ANALYSIS_REPORT.md](docs/DETAILED_QUALITY_ANALYSIS_REPORT.md)
3. Data: [data_analysis/results/summary_enhanced.json](data_analysis/results/summary_enhanced.json)

### For Developers
1. Quick start: [QUICKSTART.md](docs/QUICKSTART.md)
2. Usage: [SENTENCE_LEVEL_USAGE.md](docs/SENTENCE_LEVEL_USAGE.md)
3. Code: [scripts/](scripts/)
4. Reference: [CHEATSHEET.md](docs/CHEATSHEET.md)

### For Data Scientists
1. Overview: [README.md](docs/README.md)
2. Methodology: [METHODOLOGY_AND_DISCUSSION.md](docs/METHODOLOGY_AND_DISCUSSION.md)
3. Visualizations: [VISUALIZATION_REPORT.md](docs/VISUALIZATION_REPORT.md)
4. Results: [data_analysis/results/](data_analysis/results/)

---

## 🗑️ Temp Files (`temp_files/`)

Contains:
- **analysis.log** - Analysis execution log
- **analysis_fixed.log** - Fixed analysis log
- **temp_notebook.txt** - Temporary notebook
- **data_quality_research.ipynb** - Jupyter notebook

These can be safely deleted if not needed.

---

## ⚙️ Setup Files (`setup/`)

Contains:
- **setup.sh** - Linux/Mac setup script
- **setup.bat** - Windows setup script

Run once before first use:
```bash
bash setup/setup.sh  # On Linux/Mac
setup/setup.bat     # On Windows
```

---

## 📊 Data Model

### Error Classification (4 Categories × 12 Types)

**Generation Errors:**
- Under-generation (target < 0.6 × source)
- Over-generation (target > 1.5 × source)
- Repetition (same word/phrase > 2×)
- Off-target language (> 60% English)

**Linguistic Issues:**
- Orthographic inconsistencies
- Script mixing
- Translation artifacts

**Cohesion Errors:**
- Content errors
- Grammatical errors
- Lexical cohesion errors

**Data Noise:**
- Segmentation errors
- Annotation inconsistencies

---

## 📝 File Statistics

- **Total Markdown Files:** 16 (in docs/)
- **Python Scripts:** 10 (in scripts/)
- **Data Files:** 10 JSON files (in data_analysis/results/)
- **Visualizations:** 7 PNG images (in data_analysis/results/visualizations/)
- **Datasets:** 6 CSV files (Health/ and Tech/ folders)
- **Total Size:** ~4 GB (includes virtual environment)

---

## 🔗 Related Resources

- **Model:** facebook/nllb-200-distilled-600M (600M parameters, 200+ languages)
- **Dataset:** AfriDocMT (Masakhane community)
- **Source Languages:** English
- **Target Languages:** 5 African languages
- **Framework:** PyTorch + Transformers

---

## ✅ Maintenance

### Keep Clean
- Delete temp_files/ contents periodically
- Archive old results if needed
- Update docs as project evolves

### Backup Important
- docs/ - Documentation
- scripts/ - Source code
- data_analysis/results/data/ - JSON results

### Version Control
- .git/ - All changes tracked
- Use git for version management

---

## 🆘 Troubleshooting

**Issue:** Scripts not found
- **Solution:** Ensure you're in `data-quality-research` directory
- Run: `pwd` to check current directory

**Issue:** Import errors
- **Solution:** Install requirements: `pip install -r requirements.txt`
- Verify venv is activated: `which python`

**Issue:** Model download errors
- **Solution:** Check internet connection
- First run may take time downloading model
- Model cached after first download

**Issue:** Out of memory
- **Solution:** Reduce batch size in config
- Or run on GPU machine

---

## 📞 Contact & Support

For questions about:
- **Project:** See [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)
- **Methodology:** See [METHODOLOGY_AND_DISCUSSION.md](docs/METHODOLOGY_AND_DISCUSSION.md)
- **Code:** See [SENTENCE_LEVEL_USAGE.md](docs/SENTENCE_LEVEL_USAGE.md)
- **Results:** See [DETAILED_QUALITY_ANALYSIS_REPORT.md](docs/DETAILED_QUALITY_ANALYSIS_REPORT.md)

---

## 📅 Version History

| Date | Changes |
|------|---------|
| Jan 16, 2026 | Directory reorganized - Markdown to docs/, Scripts to scripts/, Results consolidated |
| Jan 16, 2026 | Visualizations fixed - Now populated with actual data |
| Jan 16, 2026 | INDEX.md created for navigation |
| Earlier | Initial analysis and research |

---

**Happy analyzing! 🎉**

*For the most up-to-date information, always refer to the docs/ folder.*
