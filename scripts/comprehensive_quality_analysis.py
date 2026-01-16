#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Data Quality Analysis with Visualizations

Runs complete sentence-level analysis across all African languages and domains,
generates detailed visualizations and produces easy-to-interpret reports.

Usage:
    python comprehensive_quality_analysis.py
    python comprehensive_quality_analysis.py --domains health tech --output results_comprehensive
    python comprehensive_quality_analysis.py --languages sw yo am
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

# Import project modules
from sentence_level_error import DataQualityChecker, TaskType
from config import dataset_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class ComprehensiveQualityAnalyzer:
    """Comprehensive data quality analysis with visualizations"""
    
    def __init__(self, output_dir: str = "results_comprehensive"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizations_dir = self.output_dir / "visualizations"
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Language configurations
        self.languages = {
            "am": {"name": "Amharic", "code": "amh_Ethi"},
            "ha": {"name": "Hausa", "code": "hau_Latn"},
            "sw": {"name": "Swahili", "code": "swh_Latn"},
            "yo": {"name": "Yoruba", "code": "yor_Latn"},
            "zu": {"name": "Zulu", "code": "zul_Latn"}
        }
        
        self.domains = ["health", "tech"]
        self.results = {}
        
        logger.info(f"Initialized analyzer. Output: {self.output_dir}")
    
    def load_csv_data(self, domain: str, split: str = "test") -> pd.DataFrame:
        """Load sentence-level data from local CSV"""
        csv_path = Path(domain.capitalize()) / f"{split}.csv"
        
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading {domain}/{split} from {csv_path}...")
        df = pd.read_csv(csv_path)
        logger.info(f"  Loaded {len(df)} samples")
        
        return df
    
    def convert_csv_to_mt_dataset(self, df: pd.DataFrame, target_lang: str) -> List[Dict]:
        """Convert CSV DataFrame to MT dataset format"""
        dataset = []
        
        # CSV typically has columns: en, <lang_code>
        for idx, row in df.iterrows():
            if target_lang in df.columns and pd.notna(row[target_lang]) and pd.notna(row.get('en')):
                dataset.append({
                    "source": str(row.get('en', '')),
                    "target": str(row[target_lang]),
                    "example_id": idx
                })
        
        return dataset
    
    def run_language_analysis(self, domain: str, lang_code: str, model_name: str) -> Dict:
        """Run analysis for a specific language and domain"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing {domain.upper()} domain - {self.languages[lang_code]['name']} ({lang_code})")
        logger.info(f"{'='*80}")
        
        try:
            # Load data
            df = self.load_csv_data(domain, split="test")
            if df.empty:
                return {"error": "No data loaded"}
            
            # Convert to MT dataset
            dataset = self.convert_csv_to_mt_dataset(df, lang_code)
            if not dataset:
                logger.warning(f"No valid translation pairs found for {lang_code}")
                return {"error": "No translation pairs"}
            
            logger.info(f"Processing {len(dataset)} translation pairs...")
            
            # Initialize checker
            checker = DataQualityChecker(model_name, TaskType.MT)
            
            # Detect errors
            errors = checker.check_all_errors(dataset)
            
            # Aggregate results
            error_counts = defaultdict(int)
            error_confidences = defaultdict(list)
            
            for error in errors:
                error_counts[error.error_type] += 1
                error_confidences[error.error_type].append(error.confidence)
            
            # Calculate statistics
            stats = {
                "total_samples": len(dataset),
                "total_errors": len(errors),
                "error_rate": len(errors) / len(dataset) * 100 if dataset else 0,
                "error_distribution": dict(error_counts),
                "error_confidences": {
                    k: {
                        "mean": float(np.mean(v)) if v else 0,
                        "std": float(np.std(v)) if v else 0,
                        "min": float(np.min(v)) if v else 0,
                        "max": float(np.max(v)) if v else 0
                    }
                    for k, v in error_confidences.items()
                },
                "errors": [
                    {
                        "type": e.error_type,
                        "confidence": float(e.confidence),
                        "source": e.evidence.get("source", "")[:100],
                        "target": e.evidence.get("target", "")[:100],
                        "details": e.details
                    }
                    for e in errors[:20]  # Top 20 errors
                ]
            }
            
            logger.info(f"✓ Completed: {stats['total_errors']} errors found in {len(dataset)} samples")
            logger.info(f"  Error rate: {stats['error_rate']:.2f}%")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing {domain}/{lang_code}: {e}")
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
                result_file = self.data_dir / f"{key}_results.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                logger.info(f"  Saved to {result_file}")
    
    def create_error_distribution_visualization(self):
        """Create error distribution charts across languages and domains"""
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Data Quality: Error Distribution Analysis', fontsize=16, fontweight='bold')
        
        for domain_idx, domain in enumerate(self.domains):
            ax = axes[domain_idx]
            
            # Prepare data
            error_data = defaultdict(lambda: defaultdict(int))
            
            for lang_code, stats in self.results.get(domain, {}).items():
                if "error_distribution" in stats:
                    for error_type, count in stats["error_distribution"].items():
                        error_data[error_type][self.languages[lang_code]["name"]] = count
            
            # Convert to DataFrame for plotting
            df_errors = pd.DataFrame(error_data).fillna(0)
            
            if not df_errors.empty:
                df_errors.plot(kind='bar', ax=ax, width=0.8)
                ax.set_title(f'{domain.upper()} Domain - Error Type Distribution', fontsize=12, fontweight='bold')
                ax.set_xlabel('Language', fontsize=10)
                ax.set_ylabel('Error Count', fontsize=10)
                ax.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "01_error_distribution.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_error_rate_comparison(self):
        """Create error rate comparison across languages and domains"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        data_for_plot = []
        
        for domain in self.domains:
            for lang_code, stats in self.results.get(domain, {}).items():
                if "error_rate" in stats:
                    data_for_plot.append({
                        "Domain": domain.upper(),
                        "Language": self.languages[lang_code]["name"],
                        "Error Rate (%)": stats["error_rate"],
                        "Total Samples": stats.get("total_samples", 0),
                        "Error Count": stats.get("total_errors", 0)
                    })
        
        if data_for_plot:
            df_rates = pd.DataFrame(data_for_plot)
            
            # Create grouped bar chart
            pivot_df = df_rates.pivot(index='Language', columns='Domain', values='Error Rate (%)')
            pivot_df.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], width=0.7)
            
            ax.set_title('Data Quality: Error Rate Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Language', fontsize=11)
            ax.set_ylabel('Error Rate (%)', fontsize=11)
            ax.legend(title='Domain', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', fontsize=9)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "02_error_rate_comparison.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_confidence_analysis(self):
        """Create confidence score analysis visualization"""
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Data Quality: Error Confidence Analysis', fontsize=16, fontweight='bold')
        
        for domain_idx, domain in enumerate(self.domains):
            ax = axes[domain_idx]
            
            # Prepare data
            confidence_data = []
            
            for lang_code, stats in self.results.get(domain, {}).items():
                if "error_confidences" in stats:
                    for error_type, conf_stats in stats["error_confidences"].items():
                        confidence_data.append({
                            "Language": self.languages[lang_code]["name"],
                            "Error Type": error_type,
                            "Avg Confidence": conf_stats["mean"]
                        })
            
            if confidence_data:
                df_conf = pd.DataFrame(confidence_data)
                
                # Create grouped bar chart
                pivot_conf = df_conf.pivot_table(
                    index='Error Type',
                    columns='Language',
                    values='Avg Confidence',
                    aggfunc='mean'
                )
                
                pivot_conf.plot(kind='bar', ax=ax, width=0.7)
                ax.set_title(f'{domain.upper()} Domain - Average Error Confidence by Type', fontsize=12, fontweight='bold')
                ax.set_xlabel('Error Type', fontsize=10)
                ax.set_ylabel('Confidence Score', fontsize=10)
                ax.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "03_confidence_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_sample_size_impact(self):
        """Visualize sample size and total error counts"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Data Quality: Sample Analysis', fontsize=14, fontweight='bold')
        
        # Prepare data
        data_for_plot = []
        
        for domain in self.domains:
            for lang_code, stats in self.results.get(domain, {}).items():
                data_for_plot.append({
                    "Domain": domain.upper(),
                    "Language": self.languages[lang_code]["name"],
                    "Total Samples": stats.get("total_samples", 0),
                    "Total Errors": stats.get("total_errors", 0)
                })
        
        if data_for_plot:
            df_samples = pd.DataFrame(data_for_plot)
            
            # Plot 1: Total samples
            pivot_samples = df_samples.pivot(index='Language', columns='Domain', values='Total Samples')
            pivot_samples.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'], width=0.7)
            axes[0].set_title('Total Samples per Language', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Language', fontsize=10)
            axes[0].set_ylabel('Sample Count', fontsize=10)
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', alpha=0.3)
            
            # Plot 2: Total errors
            pivot_errors = df_samples.pivot(index='Language', columns='Domain', values='Total Errors')
            pivot_errors.plot(kind='bar', ax=axes[1], color=['#FF6B6B', '#4ECDC4'], width=0.7)
            axes[1].set_title('Total Errors Detected per Language', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Language', fontsize=10)
            axes[1].set_ylabel('Error Count', fontsize=10)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.visualizations_dir / "04_sample_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def create_heatmap_error_rates(self):
        """Create heatmap of error rates across languages and domains"""
        
        # Prepare data
        heatmap_data = {}
        
        for domain in self.domains:
            heatmap_data[domain] = {}
            for lang_code, stats in self.results.get(domain, {}).items():
                heatmap_data[domain][self.languages[lang_code]["name"]] = stats.get("error_rate", 0)
        
        if heatmap_data:
            df_heatmap = pd.DataFrame(heatmap_data).T
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                df_heatmap,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn_r',
                cbar_kws={'label': 'Error Rate (%)'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )
            
            ax.set_title('Data Quality: Error Rate Heatmap (%) Across Languages and Domains', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Language', fontsize=10)
            ax.set_ylabel('Domain', fontsize=10)
            
            plt.tight_layout()
            viz_path = self.visualizations_dir / "05_error_rate_heatmap.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved visualization: {viz_path}")
            plt.close()
    
    def create_comprehensive_report(self):
        """Generate comprehensive text report"""
        
        report_path = self.reports_dir / "comprehensive_analysis_report.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*100 + "\n")
            f.write("COMPREHENSIVE DATA QUALITY ANALYSIS REPORT\n")
            f.write("="*100 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"Model: facebook/nllb-200-distilled-600M\n")
            f.write(f"Analysis Type: Sentence-Level Quality Check (Full Dataset)\n")
            f.write(f"Domains: {', '.join(d.upper() for d in self.domains)}\n")
            f.write(f"Languages: {', '.join(lang_info['name'] for lang_info in self.languages.values())}\n\n")
            
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
            f.write(f"Overall Error Rate: {overall_error_rate:.2f}%\n\n")
            
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
                
                # Language breakdown
                f.write(f"Language Breakdown:\n")
                f.write("-"*100 + "\n")
                f.write(f"{'Language':<20} {'Samples':<15} {'Errors':<15} {'Error Rate':<15} {'Top Error Type':<20}\n")
                f.write("-"*100 + "\n")
                
                for lang_code, stats in sorted(domain_results.items()):
                    if "error" not in stats:
                        lang_name = self.languages[lang_code]["name"]
                        samples = stats.get("total_samples", 0)
                        errors = stats.get("total_errors", 0)
                        error_rate = stats.get("error_rate", 0)
                        
                        # Find top error type
                        error_dist = stats.get("error_distribution", {})
                        top_error = max(error_dist.items(), key=lambda x: x[1])[0] if error_dist else "N/A"
                        
                        f.write(f"{lang_name:<20} {samples:<15,} {errors:<15} {error_rate:<14.2f}% {top_error:<20}\n")
                
                f.write("\n")
                
                # Error type analysis
                f.write(f"Error Type Distribution:\n")
                f.write("-"*100 + "\n")
                
                all_error_types = set()
                for stats in domain_results.values():
                    all_error_types.update(stats.get("error_distribution", {}).keys())
                
                for error_type in sorted(all_error_types):
                    f.write(f"\n{error_type.upper()}:\n")
                    f.write(f"{'Language':<20} {'Count':<15} {'Avg Confidence':<15} {'Std Dev':<15}\n")
                    f.write("-"*65 + "\n")
                    
                    for lang_code, stats in sorted(domain_results.items()):
                        if "error" not in stats:
                            lang_name = self.languages[lang_code]["name"]
                            error_dist = stats.get("error_distribution", {})
                            error_count = error_dist.get(error_type, 0)
                            
                            conf_stats = stats.get("error_confidences", {}).get(error_type, {})
                            avg_conf = conf_stats.get("mean", 0)
                            std_dev = conf_stats.get("std", 0)
                            
                            f.write(f"{lang_name:<20} {error_count:<15} {avg_conf:<14.3f} {std_dev:<14.3f}\n")
            
            # Key findings
            f.write(f"\n{'='*100}\n")
            f.write("KEY FINDINGS & INSIGHTS\n")
            f.write(f"{'='*100}\n\n")
            
            # Language rankings
            f.write("Language Quality Ranking (by Error Rate, lowest to highest):\n")
            lang_rankings = []
            for domain in self.domains:
                for lang_code, stats in self.results.get(domain, {}).items():
                    if "error_rate" in stats:
                        lang_rankings.append((
                            self.languages[lang_code]["name"],
                            domain,
                            stats["error_rate"]
                        ))
            
            lang_rankings.sort(key=lambda x: x[2])
            for rank, (lang, domain, rate) in enumerate(lang_rankings, 1):
                f.write(f"  {rank}. {lang:<15} ({domain.upper():<6}): {rate:6.2f}% error rate\n")
            
            # Domain comparison
            f.write(f"\nDomain Comparison:\n")
            for domain in self.domains:
                domain_results = self.results.get(domain, {})
                total_errors = sum(s.get("total_errors", 0) for s in domain_results.values())
                total_samples = sum(s.get("total_samples", 0) for s in domain_results.values())
                avg_error_rate = (total_errors / total_samples * 100) if total_samples > 0 else 0
                f.write(f"  {domain.upper()}: {avg_error_rate:.2f}% average error rate\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")
        
        logger.info(f"✓ Comprehensive report saved: {report_path}")
        return report_path
    
    def create_all_visualizations(self):
        """Generate all visualizations"""
        logger.info("\nGenerating visualizations...")
        
        self.create_error_distribution_visualization()
        self.create_error_rate_comparison()
        self.create_confidence_analysis()
        self.create_sample_size_impact()
        self.create_heatmap_error_rates()
        
        logger.info(f"✓ All visualizations saved to {self.visualizations_dir}")
    
    def save_summary_json(self):
        """Save summary as JSON"""
        summary_path = self.output_dir / "summary.json"
        
        # Create summary structure
        summary = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": "facebook/nllb-200-distilled-600M",
                "analysis_type": "sentence_level_quality_check"
            },
            "results": self.results
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Summary saved: {summary_path}")
    
    def run(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """Run complete analysis pipeline"""
        
        logger.info(f"\n{'='*100}")
        logger.info("COMPREHENSIVE DATA QUALITY ANALYSIS - FULL PIPELINE")
        logger.info(f"{'='*100}\n")
        
        # Run analyses
        self.run_all_analyses(model_name)
        
        # Generate visualizations
        self.create_all_visualizations()
        
        # Generate reports
        self.create_comprehensive_report()
        
        # Save summary
        self.save_summary_json()
        
        logger.info(f"\n{'='*100}")
        logger.info("ANALYSIS COMPLETE!")
        logger.info(f"{'='*100}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"  - Visualizations: {self.visualizations_dir}")
        logger.info(f"  - Reports: {self.reports_dir}")
        logger.info(f"  - Data: {self.data_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Data Quality Analysis with Visualizations"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["health", "tech"],
        help="Domains to analyze (default: health tech)"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["am", "ha", "sw", "yo", "zu"],
        help="Language codes to analyze (default: all)"
    )
    parser.add_argument(
        "--output",
        default="results_comprehensive",
        help="Output directory (default: results_comprehensive)"
    )
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="Model name (default: facebook/nllb-200-distilled-600M)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = ComprehensiveQualityAnalyzer(output_dir=args.output)
    analyzer.run(model_name=args.model)


if __name__ == "__main__":
    main()
