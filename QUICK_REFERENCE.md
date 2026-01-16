# Quick Reference Card

## 🚀 Quick Start Commands

```bash
# Navigate to project
cd data-quality-research

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies (first time only)
pip install -r requirements.txt

# Run main analysis
python scripts/comprehensive_quality_analysis_enhanced.py --output data_analysis/results
```

---

## 📍 Quick Navigation

| Need | Location | Command |
|------|----------|---------|
| **Documentation** | [docs/](docs/) | See INDEX.md |
| **Python Code** | [scripts/](scripts/) | `ls scripts/` |
| **Visualizations** | [data_analysis/results/visualizations/](data_analysis/results/visualizations/) | `ls data_analysis/results/visualizations/` |
| **Results Data** | [data_analysis/results/data/](data_analysis/results/data/) | `ls data_analysis/results/data/` |
| **Datasets** | [Health/](Health/) & [Tech/](Tech/) | `ls Health/ Tech/` |

---

## 📚 Key Documents

| Document | Purpose | Link |
|----------|---------|------|
| **INDEX** | Navigation guide | [INDEX.md](INDEX.md) |
| **README** | Project overview | [docs/README.md](docs/README.md) |
| **METHODOLOGY** | Main research doc | [docs/METHODOLOGY_AND_DISCUSSION.md](docs/METHODOLOGY_AND_DISCUSSION.md) |
| **FINDINGS** | Analysis results | [docs/DETAILED_QUALITY_ANALYSIS_REPORT.md](docs/DETAILED_QUALITY_ANALYSIS_REPORT.md) |
| **VISUALIZATIONS** | Visualization guide | [docs/VISUALIZATION_REPORT.md](docs/VISUALIZATION_REPORT.md) |
| **CLEANUP** | Organization summary | [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) |

---

## 🎯 Main Visualizations

1. **11_quality_issues_heatmap.png** - Overall quality snapshot by language/split
2. **12_detailed_quality_issues.png** - Detailed error breakdown (4 categories)
3. **10_error_type_breakdown.png** - Top error types ranked
4. **07_error_types_detailed.png** - Error distribution by language

---

## 🔧 Key Python Files

- **comprehensive_quality_analysis_enhanced.py** - Main analysis script
- **enhanced_error_detection.py** - Error detection engine (12 types)
- **sentence_level_error.py** - Model-based detection
- **config.py** - Configuration settings
- **utils.py** - Utility functions

---

## 📊 Results Files

- **health_*_results_enhanced.json** - Results for health domain (5 languages)
- **tech_*_results_enhanced.json** - Results for tech domain (5 languages)
- **summary_enhanced.json** - Complete summary of all results
- **comprehensive_analysis_enhanced.txt** - Text report

---

## 💡 Tips

1. **Start with INDEX.md** for complete navigation
2. **All docs in docs/** folder - never scattered in root
3. **All scripts in scripts/** folder - organized by function
4. **All results in data_analysis/results/** - consolidated single location
5. **temp_files/** can be safely deleted if not needed

---

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Run: `pip install -r requirements.txt` |
| Scripts not found | Ensure in project root: `pwd` |
| Model download slow | First run downloads model, then cached |
| Out of memory | Reduce batch size in config.py |

---

## 🎉 You're All Set!

Your project is:
- ✅ **Organized** - Clear directory structure
- ✅ **Documented** - 17 comprehensive guides
- ✅ **Clean** - No duplicates or empty files
- ✅ **Ready** - Can be used and shared

**Happy analyzing!**
