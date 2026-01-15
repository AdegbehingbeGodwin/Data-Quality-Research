#!/usr/bin/env python
"""
Compare error detection results across languages or configurations
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

def load_error_file(filepath: str) -> List[Dict]:
    """Load errors from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_errors(errors: List[Dict]) -> Dict:
    """Analyze error distribution"""
    stats = {
        "total_errors": len(errors),
        "by_type": defaultdict(int),
        "avg_confidence": defaultdict(list)
    }
    
    for error in errors:
        error_type = error["error_type"]
        stats["by_type"][error_type] += 1
        stats["avg_confidence"][error_type].append(error["confidence"])
    
    # Calculate averages
    for error_type, confidences in stats["avg_confidence"].items():
        stats["avg_confidence"][error_type] = sum(confidences) / len(confidences)
    
    return stats

def compare_languages(results_dir: str, config: str):
    """Compare error rates across languages for a given config"""
    print(f"\n{'='*80}")
    print(f"LANGUAGE COMPARISON: {config.upper()}")
    print(f"{'='*80}\n")
    
    languages = ["am", "ha", "sw", "yo", "zu"]
    lang_stats = {}
    
    for lang in languages:
        error_file = os.path.join(results_dir, config, f"errors_{config}_{lang}.json")
        
        if os.path.exists(error_file):
            errors = load_error_file(error_file)
            lang_stats[lang] = analyze_errors(errors)
        else:
            print(f"⚠️  File not found: {error_file}")
    
    if not lang_stats:
        print("No data found!")
        return
    
    # Print comparison table
    print(f"{'Language':<10} {'Total Errors':<15} {'Error Types':<15}")
    print("-" * 80)
    
    for lang, stats in sorted(lang_stats.items()):
        print(f"{lang.upper():<10} {stats['total_errors']:<15} {len(stats['by_type']):<15}")
    
    # Error type distribution
    print(f"\n{'Error Type Distribution':^80}")
    print("-" * 80)
    
    all_error_types = set()
    for stats in lang_stats.values():
        all_error_types.update(stats["by_type"].keys())
    
    print(f"{'Error Type':<30}", end="")
    for lang in sorted(lang_stats.keys()):
        print(f"{lang.upper():>10}", end="")
    print()
    print("-" * 80)
    
    for error_type in sorted(all_error_types):
        print(f"{error_type:<30}", end="")
        for lang in sorted(lang_stats.keys()):
            count = lang_stats[lang]["by_type"].get(error_type, 0)
            print(f"{count:>10}", end="")
        print()
    
    # Average confidence
    print(f"\n{'Average Confidence by Error Type':^80}")
    print("-" * 80)
    
    print(f"{'Error Type':<30}", end="")
    for lang in sorted(lang_stats.keys()):
        print(f"{lang.upper():>10}", end="")
    print()
    print("-" * 80)
    
    for error_type in sorted(all_error_types):
        print(f"{error_type:<30}", end="")
        for lang in sorted(lang_stats.keys()):
            conf = lang_stats[lang]["avg_confidence"].get(error_type, 0)
            print(f"{conf:>10.3f}", end="")
        print()

def compare_configs(results_dir: str, lang: str):
    """Compare error rates across configs for a given language"""
    print(f"\n{'='*80}")
    print(f"CONFIG COMPARISON: {lang.upper()}")
    print(f"{'='*80}\n")
    
    configs = ["health", "tech"]
    config_stats = {}
    
    for config in configs:
        error_file = os.path.join(results_dir, config, f"errors_{config}_{lang}.json")
        
        if os.path.exists(error_file):
            errors = load_error_file(error_file)
            config_stats[config] = analyze_errors(errors)
        else:
            print(f"⚠️  File not found: {error_file}")
    
    if not config_stats:
        print("No data found!")
        return
    
    # Print comparison
    print(f"{'Config':<10} {'Total Errors':<15} {'Error Types':<15}")
    print("-" * 80)
    
    for config, stats in sorted(config_stats.items()):
        print(f"{config.upper():<10} {stats['total_errors']:<15} {len(stats['by_type']):<15}")
    
    # Detailed comparison
    all_error_types = set()
    for stats in config_stats.values():
        all_error_types.update(stats["by_type"].keys())
    
    print(f"\n{'Error Type':<30} {'Health':<15} {'Tech':<15} {'Difference':<15}")
    print("-" * 80)
    
    for error_type in sorted(all_error_types):
        health_count = config_stats.get("health", {}).get("by_type", {}).get(error_type, 0)
        tech_count = config_stats.get("tech", {}).get("by_type", {}).get(error_type, 0)
        diff = tech_count - health_count
        
        print(f"{error_type:<30} {health_count:<15} {tech_count:<15} {diff:+<15}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare quality analysis results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="quality_reports",
        help="Directory containing results"
    )
    parser.add_argument(
        "--compare",
        type=str,
        choices=["languages", "configs", "both"],
        default="both",
        help="What to compare"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="health",
        help="Config for language comparison"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="sw",
        help="Language for config comparison"
    )
    
    args = parser.parse_args()
    
    if args.compare in ["languages", "both"]:
        compare_languages(args.results_dir, args.config)
    
    if args.compare in ["configs", "both"]:
        compare_configs(args.results_dir, args.lang)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
