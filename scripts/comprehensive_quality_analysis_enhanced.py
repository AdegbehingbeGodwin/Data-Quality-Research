#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Data Quality Analysis - ENHANCED VERSION

Combines ALL data splits (train, dev, test) for complete analysis
Tracks data split origin and visualizes quality across splits
Includes detailed error categorization and split-wise analysis

Usage:
    python comprehensive_quality_analysis_enhanced.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from collections import defaultdict
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from sentence_level_error import DataQualityChecker, TaskType
from config import dataset_config
from enhanced_error_detection import categorize_errors_comprehensively, DetailedQualityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class EnhancedQualityAnalyzer:
    """Enhanced analyzer combining all data splits"""
    
    def __init__(self, output_dir: str = "results_enhanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizations_dir = self.output_dir / "visualizations"
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.languages = {
            "am": {"name": "Amharic", "code": "amh_Ethi"},
            "ha": {"name": "Hausa", "code": "hau_Latn"},
            "sw": {"name": "Swahili", "code": "swh_Latn"},
            "yo": {"name": "Yoruba", "code": "yor_Latn"},
            "zu": {"name": "Zulu", "code": "zul_Latn"}
        }
        
        self.domains = ["health", "tech"]
        self.splits = ["train", "dev", "test"]
        self.results = {}
        
        logger.info(f"Initialized enhanced analyzer. Output: {self.output_dir}")
    
    def load_all_splits(self, domain: str, target_lang: str) -> Tuple[List[Dict], Dict]:
        """Load and combine all splits (train, dev, test)"""
        
        all_data = []
        split_counts = {"train": 0, "dev": 0, "test": 0}
        
        for split in self.splits:
            csv_path = Path(domain.capitalize()) / f"{split}.csv"
            
            if not csv_path.exists():
                logger.warning(f"CSV not found: {csv_path}")
                continue
            
            logger.info(f"Loading {domain}/{split}...")
            df = pd.read_csv(csv_path)
            
            # Convert to MT dataset format
            for idx, row in df.iterrows():
                if target_lang in df.columns and pd.notna(row[target_lang]) and pd.notna(row.get('en')):
                    all_data.append({
                        "source": str(row.get('en', '')),
                        "target": str(row[target_lang]),
                        "split": split,
                        "example_id": f"{split}_{idx}"
                    })
                    split_counts[split] += 1
        
        total = sum(split_counts.values())
        logger.info(f"  Combined samples: {total} (train: {split_counts['train']}, dev: {split_counts['dev']}, test: {split_counts['test']})")
        
        return all_data, split_counts
    
    def run_language_analysis(self, domain: str, lang_code: str, model_name: str) -> Dict:
        """Run analysis for all splits of a language/domain"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing {domain.upper()} - {self.languages[lang_code]['name']} (ALL SPLITS)")
        logger.info(f"{'='*80}")
        
        try:
            # Load all data
            dataset, split_counts = self.load_all_splits(domain, lang_code)
            if not dataset:
                return {"error": "No translation pairs"}
            
            logger.info(f"Running enhanced error detection...")
            
            # Run comprehensive detailed error analysis
            analyzer = DetailedQualityAnalyzer()
            detailed_analysis = analyzer.analyze_all_issues(dataset)
            
            # Initialize checker for model-based detection
            checker = DataQualityChecker(model_name, TaskType.MT)
            
            # Detect model-based errors
            logger.info(f"Processing {len(dataset)} translation pairs with model...")
            errors = checker.check_all_errors(dataset)
            
            # Aggregate results by split - using dataset to map errors back to splits
            error_counts_by_split = defaultdict(lambda: defaultdict(int))
            error_confidences_by_split = defaultdict(lambda: defaultdict(list))
            error_counts_total = defaultdict(int)
            error_confidences_total = defaultdict(list)
            
            # Create a mapping of source text to split for error attribution
            sample_to_split = {item['source']: item['split'] for item in dataset}
            
            for error in errors:
                # Try to get split from evidence source
                split = "unknown"
                source = error.evidence.get("source", "")
                
                if source and source in sample_to_split:
                    split = sample_to_split[source]
                
                error_counts_by_split[split][error.error_type] += 1
                error_counts_total[error.error_type] += 1
                error_confidences_by_split[split][error.error_type].append(error.confidence)
                error_confidences_total[error.error_type].append(error.confidence)
            
            # Calculate statistics per split
            split_stats = {}
            for split in self.splits:
                split_total = len([d for d in dataset if d["split"] == split])
                split_errors = sum(error_counts_by_split[split].values())
                split_stats[split] = {
                    "total_samples": split_total,
                    "total_errors": split_errors,
                    "error_rate": (split_errors / split_total * 100) if split_total > 0 else 0,
                    "error_distribution": dict(error_counts_by_split[split])
                }
            
            # Overall statistics
            stats = {
                "total_samples": len(dataset),
                "total_errors": len(errors),
                "error_rate": len(errors) / len(dataset) * 100 if dataset else 0,
                "split_counts": split_counts,
                "split_stats": split_stats,
                "error_distribution": dict(error_counts_total),
                "detailed_quality_issues": detailed_analysis["summary"],
                "quality_issues_details": {
                    "generation_errors": {
                        "under_generation": len(detailed_analysis["detailed_errors"]["generation_errors"]["under_generation"]),
                        "over_generation": len(detailed_analysis["detailed_errors"]["generation_errors"]["over_generation"]),
                        "repetition": len(detailed_analysis["detailed_errors"]["generation_errors"]["repetition"]),
                        "off_target_language": len(detailed_analysis["detailed_errors"]["generation_errors"]["off_target_language"])
                    },
                    "linguistic_issues": {
                        "orthographic_inconsistencies": len(detailed_analysis["detailed_errors"]["linguistic_issues"]["orthographic_inconsistencies"]),
                        "script_mixing": len(detailed_analysis["detailed_errors"]["linguistic_issues"]["script_mixing"]),
                        "translation_artifacts": len(detailed_analysis["detailed_errors"]["linguistic_issues"]["translation_artifacts"])
                    },
                    "cohesion_errors": {
                        "content_errors": len(detailed_analysis["detailed_errors"]["cohesion_errors"]["content_errors"]),
                        "grammatical_errors": len(detailed_analysis["detailed_errors"]["cohesion_errors"]["grammatical_errors"]),
                        "lexical_cohesion_errors": len(detailed_analysis["detailed_errors"]["cohesion_errors"]["lexical_cohesion_errors"])
                    },
                    "data_noise": {
                        "segmentation_errors": len(detailed_analysis["detailed_errors"]["data_noise"]["segmentation_errors"]),
                        "annotation_inconsistencies": len(detailed_analysis["detailed_errors"]["data_noise"]["annotation_inconsistencies"])
                    }
                },
                "error_confidences": {
                    k: {
                        "mean": float(np.mean(v)) if v else 0,
                        "std": float(np.std(v)) if v else 0,
                        "min": float(np.min(v)) if v else 0,
                        "max": float(np.max(v)) if v else 0,
                        "count": len(v)
                    }
                    for k, v in error_confidences_total.items()
                },
                "errors": [
                    {
                        "type": e.error_type,
                        "confidence": float(e.confidence),
                        "split": sample_to_split.get(e.evidence.get("source", ""), "unknown"),
                        "source": e.evidence.get("source", "")[:100],
                        "target": e.evidence.get("target", "")[:100]
                    }
                    for e in errors[:30]
                ]
            }
            
            logger.info(f"✓ Completed: {stats['total_errors']} model-detected errors in {len(dataset)} samples")
            logger.info(f"  Overall error rate: {stats['error_rate']:.2f}%")
            for split in self.splits:
                logger.info(f"  {split.upper()}: {split_stats[split]['error_rate']:.2f}% ({split_stats[split]['total_samples']} samples)")
            
            # Log detailed error detection results
            logger.info(f"\n  DETAILED QUALITY ISSUES:")
            logger.info(f"    Generation Errors:")
            logger.info(f"      Under-generation: {stats['quality_issues_details']['generation_errors']['under_generation']}")
            logger.info(f"      Over-generation: {stats['quality_issues_details']['generation_errors']['over_generation']}")
            logger.info(f"      Repetition: {stats['quality_issues_details']['generation_errors']['repetition']}")
            logger.info(f"      Off-target language: {stats['quality_issues_details']['generation_errors']['off_target_language']}")
            logger.info(f"    Linguistic Issues:")
            logger.info(f"      Orthographic inconsistencies: {stats['quality_issues_details']['linguistic_issues']['orthographic_inconsistencies']}")
            logger.info(f"      Script mixing: {stats['quality_issues_details']['linguistic_issues']['script_mixing']}")
            logger.info(f"      Translation artifacts: {stats['quality_issues_details']['linguistic_issues']['translation_artifacts']}")
            logger.info(f"    Cohesion Errors:")
            logger.info(f"      Content errors: {stats['quality_issues_details']['cohesion_errors']['content_errors']}")
            logger.info(f"      Grammatical errors: {stats['quality_issues_details']['cohesion_errors']['grammatical_errors']}")
            logger.info(f"      Lexical cohesion errors: {stats['quality_issues_details']['cohesion_errors']['lexical_cohesion_errors']}")
            logger.info(f"    Data Preparation Noise:")
            logger.info(f"      Segmentation errors: {stats['quality_issues_details']['data_noise']['segmentation_errors']}")
            logger.info(f"      Annotation inconsistencies: {stats['quality_issues_details']['data_noise']['annotation_inconsistencies']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing {domain}/{lang_code}: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def run_all_analyses(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """Run analysis for all languages and domains"""
        
        for domain in self.domains:
            self.results[domain] = {}
            
            for lang_code in self.languages.keys():
                key = f"{domain}_{lang_code}"
                stats = self.run_language_analysis(domain, lang_code, model_name)
                self.results[domain][lang_code] = stats
                
                # Save individual results
                result_file = self.data_dir / f"{key}_results_enhanced.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                logger.info(f"  Saved to {result_file}")
    
    def create_split_comparison_viz(self):
        """Visualize quality across data splits"""
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Data Quality Across Data Splits (Train/Dev/Test)', fontsize=16, fontweight='bold')
        
        for domain_idx, domain in enumerate(self.domains):
            ax = axes[domain_idx]
            
            # Prepare data
            split_data = []
            
            for lang_code, stats in self.results.get(domain, {}).items():
                if "split_stats" in stats:
                    for split, split_stat in stats["split_stats"].items():
                        split_data.append({
                            "Language": self.languages[lang_code]["name"],
                            "Split": split.upper(),
                            "Error Rate (%)": split_stat["error_rate"],
                            "Samples": split_stat["total_samples"],
                            "Errors": split_stat["total_errors"]
                        })
            
            if split_data:
                df_splits = pd.DataFrame(split_data)
                
                # Create pivot for grouped bars
                pivot_df = df_splits.pivot_table(
                    index='Language',
                    columns='Split',
                    values='Error Rate (%)',
                    aggfunc='mean'
                )
                
                # Reorder splits (train, dev, test)
                pivot_df = pivot_df[['TRAIN', 'DEV', 'TEST']]
                
                pivot_df.plot(kind='bar', ax=ax, width=0.7, color=['#FF6B6B', '#FFA07A', '#4ECDC4'])
                ax.set_title(f'{domain.upper()} Domain - Error Rates by Data Split', fontsize=12, fontweight='bold')
                ax.set_xlabel('Language', fontsize=10)
                ax.set_ylabel('Error Rate (%)', fontsize=10)
                ax.legend(title='Split', fontsize=9)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "06_split_comparison.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_error_type_detailed_viz(self):
        """Detailed error type visualization"""
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Error Type Distribution Analysis', fontsize=16, fontweight='bold')
        
        for domain_idx, domain in enumerate(self.domains):
            ax = axes[domain_idx]
            
            # Prepare data - collect all error types across all languages
            error_data = {}
            
            for lang_code, stats in self.results.get(domain, {}).items():
                lang_name = self.languages[lang_code]["name"]
                errors_per_lang = {}
                
                if "detailed_quality_issues" in stats:
                    # Flatten all error categories
                    quality_issues = stats["detailed_quality_issues"]
                    for category in quality_issues.values():
                        if isinstance(category, dict):
                            for error_type, count in category.items():
                                errors_per_lang[error_type.replace("_", " ").title()] = count
                
                error_data[lang_name] = errors_per_lang
            
            # Convert to DataFrame
            df_errors = pd.DataFrame(error_data).fillna(0)
            
            if not df_errors.empty:
                df_errors.plot(kind='barh', ax=ax, width=0.8)
                ax.set_title(f'{domain.upper()} Domain - Error Types', fontsize=12, fontweight='bold')
                ax.set_xlabel('Error Count', fontsize=10)
                ax.set_ylabel('Language', fontsize=10)
                ax.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "07_error_types_detailed.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_sample_coverage_viz(self):
        """Visualize sample distribution across splits"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Sample Coverage Across Data Splits', fontsize=14, fontweight='bold')
        
        for domain_idx, domain in enumerate(self.domains):
            ax = axes[domain_idx]
            
            # Prepare data
            split_samples = defaultdict(lambda: defaultdict(int))
            
            for lang_code, stats in self.results.get(domain, {}).items():
                if "split_stats" in stats:
                    for split, split_stat in stats["split_stats"].items():
                        split_samples[self.languages[lang_code]["name"]][split.upper()] = split_stat["total_samples"]
            
            if split_samples:
                df_samples = pd.DataFrame.from_dict(split_samples, orient='index')
                df_samples = df_samples[['TRAIN', 'DEV', 'TEST']]
                
                df_samples.plot(kind='bar', ax=ax, width=0.7, stacked=False, color=['#FF6B6B', '#FFA07A', '#4ECDC4'])
                ax.set_title(f'{domain.upper()} - Samples per Language', fontsize=12, fontweight='bold')
                ax.set_xlabel('Language', fontsize=10)
                ax.set_ylabel('Sample Count', fontsize=10)
                ax.legend(title='Split', fontsize=9)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "08_sample_coverage.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_split_wise_error_table_viz(self):
        """Create detailed split-wise error tables as visualizations"""
        
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))
        fig.suptitle('Error Distribution by Data Split (Train/Dev/Test)', fontsize=16, fontweight='bold', y=0.98)
        
        for domain_idx, domain in enumerate(self.domains):
            ax = axes[domain_idx]
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data table
            table_data = []
            headers = ['Language', 'TRAIN\nErrors/Samples', 'TRAIN\nError %', 
                      'DEV\nErrors/Samples', 'DEV\nError %',
                      'TEST\nErrors/Samples', 'TEST\nError %']
            
            for lang_code in sorted(self.languages.keys()):
                stats = self.results.get(domain, {}).get(lang_code, {})
                if "split_stats" in stats:
                    row = [self.languages[lang_code]["name"]]
                    
                    for split in ['train', 'dev', 'test']:
                        split_stat = stats["split_stats"].get(split, {})
                        errors = split_stat.get("total_errors", 0)
                        samples = split_stat.get("total_samples", 0)
                        error_rate = split_stat.get("error_rate", 0)
                        
                        row.append(f"{errors}/{samples}")
                        row.append(f"{error_rate:.2f}%")
                    
                    table_data.append(row)
            
            # Create table
            if table_data:
                table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', 
                                loc='center', bbox=[0, 0, 1, 1])
                
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2.5)
                
                # Style header
                for i in range(len(headers)):
                    table[(0, i)].set_facecolor('#4472C4')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                # Alternate row colors
                for i in range(1, len(table_data) + 1):
                    for j in range(len(headers)):
                        if i % 2 == 0:
                            table[(i, j)].set_facecolor('#E7E6E6')
                        else:
                            table[(i, j)].set_facecolor('#F2F2F2')
                
                ax.set_title(f'{domain.upper()} Domain', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "09_split_error_tables.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_error_type_breakdown_viz(self):
        """Create detailed error type breakdown visualization"""
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Data Quality Issues: Error Type Distribution by Language', fontsize=16, fontweight='bold')
        
        for domain_idx, domain in enumerate(self.domains):
            ax = axes[domain_idx]
            
            # Collect error types across all languages
            error_summary = {}
            
            for lang_code, stats in self.results.get(domain, {}).items():
                if "detailed_quality_issues" in stats:
                    quality_issues = stats["detailed_quality_issues"]
                    for category, errors in quality_issues.items():
                        if isinstance(errors, dict):
                            for error_type, count in errors.items():
                                clean_name = error_type.replace("_", " ").title()
                                if clean_name not in error_summary:
                                    error_summary[clean_name] = 0
                                error_summary[clean_name] += count
            
            if error_summary:
                # Sort by count
                error_summary = dict(sorted(error_summary.items(), key=lambda x: x[1], reverse=True))
                
                ax.barh(range(len(error_summary)), list(error_summary.values()), 
                       color='#FF6B6B', edgecolor='black', linewidth=0.5)
                ax.set_yticks(range(len(error_summary)))
                ax.set_yticklabels(error_summary.keys(), fontsize=9)
                ax.set_xlabel('Total Error Count', fontsize=10, fontweight='bold')
                ax.set_title(f'{domain.upper()} Domain - Top Error Types', fontsize=12, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, v in enumerate(error_summary.values()):
                    ax.text(v + 100, i, str(int(v)), va='center', fontsize=8)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "10_error_type_breakdown.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_quality_issues_heatmap(self):
        """Create heatmap showing data quality issues by language and split"""
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Data Quality Issues Heatmap: Error Rates by Language and Split', 
                    fontsize=16, fontweight='bold')
        
        for domain_idx, domain in enumerate(self.domains):
            ax = axes[domain_idx]
            
            # Prepare heatmap data
            heatmap_data = []
            languages_list = []
            
            for lang_code in sorted(self.languages.keys()):
                stats = self.results.get(domain, {}).get(lang_code, {})
                languages_list.append(self.languages[lang_code]["name"])
                
                row = []
                for split in ['train', 'dev', 'test']:
                    split_stat = stats.get("split_stats", {}).get(split, {})
                    error_rate = split_stat.get("error_rate", 0)
                    row.append(error_rate)
                
                heatmap_data.append(row)
            
            if heatmap_data:
                df_heatmap = pd.DataFrame(heatmap_data, 
                                         index=languages_list,
                                         columns=['TRAIN', 'DEV', 'TEST'])
                
                sns.heatmap(df_heatmap, annot=True, fmt='.2f', cmap='YlOrRd', 
                           ax=ax, cbar_kws={'label': 'Error Rate (%)'}, 
                           linewidths=0.5, linecolor='gray')
                ax.set_title(f'{domain.upper()} Domain', fontsize=12, fontweight='bold')
                ax.set_xlabel('Data Split', fontsize=10)
                ax.set_ylabel('Language', fontsize=10)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "11_quality_issues_heatmap.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_detailed_quality_issues_viz(self):
        """Create comprehensive visualization of all detailed quality issues"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Detailed Sentence-Level Data Quality Issues Analysis', 
                    fontsize=18, fontweight='bold')
        
        for domain_idx, domain in enumerate([self.domains[0]] + [self.domains[1]]):
            # Generation Errors
            ax1 = fig.add_subplot(gs[0, domain_idx])
            gen_errors = defaultdict(int)
            for lang_code, stats in self.results.get(domain, {}).items():
                if "quality_issues_details" in stats:
                    gen = stats["quality_issues_details"].get("generation_errors", {})
                    for error_type, count in gen.items():
                        gen_errors[error_type.replace("_", "\n")] += count
            
            if gen_errors:
                ax1.bar(range(len(gen_errors)), list(gen_errors.values()), color='#FF6B6B')
                ax1.set_xticks(range(len(gen_errors)))
                ax1.set_xticklabels(gen_errors.keys(), fontsize=8, rotation=45, ha='right')
                ax1.set_ylabel('Count', fontsize=10)
                ax1.set_title(f'{domain.upper()} - Generation Errors', fontsize=11, fontweight='bold')
                ax1.grid(axis='y', alpha=0.3)
            
            # Linguistic Issues
            ax2 = fig.add_subplot(gs[1, domain_idx])
            ling_errors = defaultdict(int)
            for lang_code, stats in self.results.get(domain, {}).items():
                if "quality_issues_details" in stats:
                    ling = stats["quality_issues_details"].get("linguistic_issues", {})
                    for error_type, count in ling.items():
                        ling_errors[error_type.replace("_", "\n")] += count
            
            if ling_errors:
                ax2.bar(range(len(ling_errors)), list(ling_errors.values()), color='#FFA07A')
                ax2.set_xticks(range(len(ling_errors)))
                ax2.set_xticklabels(ling_errors.keys(), fontsize=8, rotation=45, ha='right')
                ax2.set_ylabel('Count', fontsize=10)
                ax2.set_title(f'{domain.upper()} - Linguistic Issues', fontsize=11, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
            
            # Cohesion + Data Noise
            ax3 = fig.add_subplot(gs[2, domain_idx])
            all_errors = defaultdict(int)
            for lang_code, stats in self.results.get(domain, {}).items():
                if "quality_issues_details" in stats:
                    cohesion = stats["quality_issues_details"].get("cohesion_errors", {})
                    noise = stats["quality_issues_details"].get("data_noise", {})
                    for error_type, count in {**cohesion, **noise}.items():
                        all_errors[error_type.replace("_", "\n")] += count
            
            if all_errors:
                ax3.barh(range(len(all_errors)), list(all_errors.values()), color='#4ECDC4')
                ax3.set_yticks(range(len(all_errors)))
                ax3.set_yticklabels(all_errors.keys(), fontsize=8)
                ax3.set_xlabel('Count', fontsize=10)
                ax3.set_title(f'{domain.upper()} - Cohesion & Data Noise', fontsize=11, fontweight='bold')
                ax3.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "12_detailed_quality_issues.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_comprehensive_report(self):
        """Generate comprehensive report with split analysis"""
        
        report_path = self.reports_dir / "comprehensive_analysis_enhanced.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*100 + "\n")
            f.write("COMPREHENSIVE DATA QUALITY ANALYSIS - ENHANCED (ALL SPLITS)\n")
            f.write("="*100 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"Model: facebook/nllb-200-distilled-600M\n")
            f.write(f"Analysis Type: Sentence-Level Quality Check (Combined Train+Dev+Test)\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*100 + "\n")
            
            total_samples = sum(
                stats.get("total_samples", 0)
                for domain_results in self.results.values()
                for stats in domain_results.values()
            )
            total_errors = sum(
                stats.get("total_errors", 0)
                for domain_results in self.results.values()
                for stats in domain_results.values()
            )
            overall_error_rate = (total_errors / total_samples * 100) if total_samples > 0 else 0
            
            f.write(f"Total Samples Analyzed: {total_samples:,}\n")
            f.write(f"Total Errors Detected: {total_errors:,}\n")
            f.write(f"Overall Error Rate: {overall_error_rate:.2f}%\n")
            f.write(f"Data Splits: Train + Dev + Test (combined)\n\n")
            
            # Domain-specific analysis
            for domain in self.domains:
                f.write(f"\n{'='*100}\n")
                f.write(f"{domain.upper()} DOMAIN ANALYSIS\n")
                f.write(f"{'='*100}\n\n")
                
                domain_results = self.results.get(domain, {})
                domain_total_samples = sum(s.get("total_samples", 0) for s in domain_results.values())
                domain_total_errors = sum(s.get("total_errors", 0) for s in domain_results.values())
                domain_error_rate = (domain_total_errors / domain_total_samples * 100) if domain_total_samples > 0 else 0
                
                f.write(f"Domain Summary:\n")
                f.write(f"  Total Samples: {domain_total_samples:,}\n")
                f.write(f"  Total Errors: {domain_total_errors:,}\n")
                f.write(f"  Error Rate: {domain_error_rate:.2f}%\n\n")
                
                # Overall language breakdown
                f.write(f"Language Breakdown (Overall):\n")
                f.write("-"*100 + "\n")
                f.write(f"{'Language':<20} {'Samples':<15} {'Errors':<15} {'Error Rate':<15}\n")
                f.write("-"*100 + "\n")
                
                for lang_code, stats in sorted(domain_results.items()):
                    if "error" not in stats:
                        lang_name = self.languages[lang_code]["name"]
                        samples = stats.get("total_samples", 0)
                        errors = stats.get("total_errors", 0)
                        error_rate = stats.get("error_rate", 0)
                        f.write(f"{lang_name:<20} {samples:<15,} {errors:<15} {error_rate:<14.2f}%\n")
                
                # Detailed split-wise breakdown per language
                f.write(f"\n\nDETAILED SPLIT-WISE ERROR ANALYSIS:\n")
                f.write("-"*100 + "\n\n")
                
                for lang_code in sorted(self.languages.keys()):
                    stats = domain_results.get(lang_code)
                    if stats and "split_stats" in stats:
                        lang_name = self.languages[lang_code]["name"]
                        f.write(f"{lang_name}:\n")
                        f.write(f"{'Split':<15} {'Samples':<15} {'Errors':<15} {'Error Rate':<15} {'Error Types':<30}\n")
                        f.write("-"*90 + "\n")
                        
                        for split in ['train', 'dev', 'test']:
                            if split in stats["split_stats"]:
                                split_stat = stats["split_stats"][split]
                                samples = split_stat.get('total_samples', 0)
                                errors = split_stat.get('total_errors', 0)
                                error_rate = split_stat.get('error_rate', 0)
                                error_types = split_stat.get('error_distribution', {})
                                
                                error_type_str = ", ".join([f"{k}({v})" for k, v in error_types.items()]) if error_types else "None"
                                
                                f.write(f"{split.upper():<15} {samples:<15,} {errors:<15} {error_rate:<14.2f}% {error_type_str:<30}\n")
                        
                        f.write("\n")
                
                # Error type summary for domain
                f.write(f"\nERROR TYPE SUMMARY FOR {domain.upper()}:\n")
                f.write("-"*100 + "\n")
                
                all_error_types = {}
                for lang_code, stats in domain_results.items():
                    if "error_distribution" in stats:
                        for error_type, count in stats["error_distribution"].items():
                            if error_type not in all_error_types:
                                all_error_types[error_type] = 0
                            all_error_types[error_type] += count
                
                for error_type, count in sorted(all_error_types.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {error_type}: {count} occurrences\n")
                
                f.write("\n")
        
        logger.info(f"✓ Comprehensive report saved: {report_path}")
        return report_path
    
    def create_all_visualizations(self):
        """Generate all visualizations"""
        logger.info("\nGenerating visualizations...")
        
        self.create_split_comparison_viz()
        self.create_error_type_detailed_viz()
        self.create_sample_coverage_viz()
        self.create_split_wise_error_table_viz()
        self.create_error_type_breakdown_viz()
        self.create_quality_issues_heatmap()
        self.create_detailed_quality_issues_viz()
        
        logger.info(f"✓ All visualizations saved to {self.visualizations_dir}")
    
    def save_summary_json(self):
        """Save summary as JSON"""
        summary_path = self.output_dir / "summary_enhanced.json"
        
        summary = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": "facebook/nllb-200-distilled-600M",
                "analysis_type": "comprehensive_with_all_splits",
                "note": "Combines train + dev + test splits"
            },
            "results": self.results
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Summary saved: {summary_path}")
    
    def run(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """Run complete analysis pipeline"""
        
        logger.info(f"\n{'='*100}")
        logger.info("COMPREHENSIVE DATA QUALITY ANALYSIS - ENHANCED")
        logger.info("COMBINING ALL DATA SPLITS (TRAIN + DEV + TEST)")
        logger.info(f"{'='*100}\n")
        
        # Run analyses
        self.run_all_analyses(model_name)
        
        # Generate visualizations
        self.create_all_visualizations()
        
        # Save summary
        self.save_summary_json()
        
        # Generate reports (wrapped in try-except to avoid blocking)
        try:
            logger.info("\nGenerating comprehensive report...")
            self.create_comprehensive_report()
        except Exception as e:
            logger.warning(f"Report generation had an issue (continuing): {e}")
        
        logger.info(f"\n{'='*100}")
        logger.info("ANALYSIS COMPLETE!")
        logger.info(f"{'='*100}")
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Comprehensive Data Quality Analysis"
    )
    parser.add_argument(
        "--output",
        default="results_enhanced",
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="Model name"
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = EnhancedQualityAnalyzer(output_dir=args.output)
    analyzer.run(model_name=args.model)


if __name__ == "__main__":
    main()
