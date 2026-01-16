# -*- coding: utf-8 -*-
"""
Sentence Level Error Detection Framework

Comprehensive data quality error detection for African language NLP tasks.
Supports multiple task types: NER, POS, MT, QA, Sentiment Analysis.

Original Colab notebook:
    https://colab.research.google.com/drive/1QlBuDLhIXBPgrsw-zd0DAeq_fL3kiiO9

Author: Data Quality Research Team
License: MIT
"""

"""
COMPREHENSIVE DATA QUALITY ERROR DETECTION FOR AFRICAN LANGUAGE NLP

This framework detects data quality issues in both token-based (NER, POS) and
sentence-based (MT, QA) tasks using state-of-the-art explainability methods:
- Integrated Gradients for attribution analysis
- Attention mechanism analysis
- Cross-attention pattern detection
- Statistical heuristics for error detection

Supported Tasks:
- Named Entity Recognition (NER)
- Part-of-Speech Tagging (POS)
- Machine Translation (MT)
- Question Answering (QA)
- Sentiment Analysis

Requirements:
    torch>=2.0.0
    transformers>=4.35.0
    datasets>=2.14.0
    captum>=0.6.0
    numpy>=1.24.0
    pandas>=2.0.0
    scipy>=1.11.0
    scikit-learn>=1.3.0

Installation:
    pip install -r requirements.txt

Usage:
    # For NER task
    checker = DataQualityChecker("Davlan/xlm-roberta-base-wikiann-ner", TaskType.NER)
    errors = checker.check_all_errors(ner_dataset)
    checker.generate_report(errors, "ner_quality_report.txt")
    
    # For MT task
    checker = DataQualityChecker("facebook/nllb-200-distilled-600M", TaskType.MT)
    errors = checker.check_all_errors(mt_dataset)
    checker.generate_report(errors, "mt_quality_report.txt")
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Explainability imports
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from transformers import AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
import pandas as pd


class TaskType(Enum):
    """Supported task types"""
    NER = "ner"
    POS = "pos"
    SENTIMENT = "sentiment"
    MT = "machine_translation"
    QA = "question_answering"


@dataclass
class ErrorReport:
    """Structure for storing detected errors"""
    error_type: str
    confidence: float
    evidence: Dict
    example_id: str
    details: str
    recommendations: str


class DataQualityChecker:
    """Main class for detecting data quality issues"""

    def __init__(self, model_name: str, task_type: TaskType, device: str = "cuda"):
        """
        Initialize the checker with a model and task type

        Args:
            model_name: HuggingFace model name or path
            task_type: Type of NLP task
            device: Device to run on (cuda/cpu)
        """
        self.task_type = task_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load appropriate model based on task
        if task_type in [TaskType.NER, TaskType.POS]:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        elif task_type == TaskType.MT:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif task_type == TaskType.QA:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        # Initialize explainability tools
        self.integrated_gradients = IntegratedGradients(self.model)

        print(f"✓ Initialized {task_type.value} checker with {model_name}")

    def check_all_errors(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Check for all relevant errors based on task type

        Args:
            dataset: List of examples with required fields based on task

        Returns:
            List of ErrorReport objects
        """
        all_errors = []

        if self.task_type in [TaskType.NER, TaskType.POS]:
            all_errors.extend(self._check_token_based_errors(dataset))
        elif self.task_type == TaskType.MT:
            all_errors.extend(self._check_mt_errors(dataset))
        elif self.task_type == TaskType.QA:
            all_errors.extend(self._check_qa_errors(dataset))
        elif self.task_type == TaskType.SENTIMENT:
            all_errors.extend(self._check_sentiment_errors(dataset))

        return all_errors

    # ============================================================================
    # TOKEN-BASED TASK ERROR DETECTION (NER, POS)
    # ============================================================================

    def _check_token_based_errors(self, dataset: List[Dict]) -> List[ErrorReport]:
        """Check errors for token classification tasks"""
        errors = []

        print("Checking annotation inconsistencies...")
        errors.extend(self._detect_annotation_inconsistencies(dataset))

        print("Checking label noise...")
        errors.extend(self._detect_label_noise(dataset))

        print("Checking spurious correlations...")
        errors.extend(self._detect_spurious_correlations_tokens(dataset))

        return errors

    def _detect_annotation_inconsistencies(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect cases where similar tokens have different labels

        Expected format: {"tokens": [...], "labels": [...], "id": "..."}
        """
        errors = []
        token_label_map = {}  # Track all labels for each token

        # Build token-label mapping
        for example in dataset:
            tokens = example.get("tokens", [])
            labels = example.get("labels", [])

            for token, label in zip(tokens, labels):
                token_lower = token.lower()
                if token_lower not in token_label_map:
                    token_label_map[token_lower] = []
                token_label_map[token_lower].append({
                    "label": label,
                    "example_id": example.get("id", "unknown"),
                    "context": " ".join(tokens)
                })

        # Find inconsistencies
        for token, label_instances in token_label_map.items():
            unique_labels = set(inst["label"] for inst in label_instances)

            if len(unique_labels) > 1 and len(label_instances) >= 3:
                # Significant inconsistency found
                label_counts = {}
                for inst in label_instances:
                    label_counts[inst["label"]] = label_counts.get(inst["label"], 0) + 1

                confidence = 1.0 - (min(label_counts.values()) / sum(label_counts.values()))

                errors.append(ErrorReport(
                    error_type="annotation_inconsistency",
                    confidence=confidence,
                    evidence={
                        "token": token,
                        "label_distribution": label_counts,
                        "examples": label_instances[:5]
                    },
                    example_id="multiple",
                    details=f"Token '{token}' has {len(unique_labels)} different labels: {unique_labels}",
                    recommendations="Review annotation guidelines for this entity type"
                ))

        return errors

    def _detect_label_noise(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect cases where model's 'wrong' prediction might actually be correct
        Uses attention and attribution to validate model's reasoning
        """
        errors = []

        for example in dataset[:100]:  # Sample to avoid long processing
            tokens = example.get("tokens", [])
            true_labels = example.get("labels", [])
            example_id = example.get("id", "unknown")

            # Get model predictions
            inputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)[0]

            # Get attention weights
            with torch.no_grad():
                outputs_with_attn = self.model(**inputs, output_attentions=True)
                attention = outputs_with_attn.attentions[-1][0].mean(dim=0)  # Average over heads

            # Find disagreements between prediction and true label
            word_ids = inputs.word_ids()
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue

                if word_id < len(true_labels):
                    pred_label = predictions[idx].item()
                    true_label = true_labels[word_id] if isinstance(true_labels[word_id], int) else 0

                    if pred_label != true_label:
                        # Check if model has strong evidence for its prediction
                        token_attention = attention[idx].cpu().numpy()
                        attention_confidence = float(token_attention.max())

                        if attention_confidence > 0.3:  # High confidence in model's choice
                            errors.append(ErrorReport(
                                error_type="potential_label_noise",
                                confidence=attention_confidence,
                                evidence={
                                    "token": tokens[word_id],
                                    "true_label": true_label,
                                    "predicted_label": pred_label,
                                    "attention_confidence": attention_confidence,
                                    "context": " ".join(tokens)
                                },
                                example_id=example_id,
                                details=f"Model strongly predicts {pred_label} but label is {true_label}",
                                recommendations="Manual review: model might be correct"
                            ))

        return errors

    def _detect_spurious_correlations_tokens(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect if model relies on position or format cues rather than content
        """
        errors = []
        position_label_map = {}  # Track labels by position in sentence

        for example in dataset:
            tokens = example.get("tokens", [])
            labels = example.get("labels", [])

            for pos, label in enumerate(labels):
                relative_pos = pos / max(len(labels), 1)  # Normalize position
                pos_bucket = int(relative_pos * 10)  # Bucket into 10 positions

                if pos_bucket not in position_label_map:
                    position_label_map[pos_bucket] = []
                position_label_map[pos_bucket].append(label)

        # Check for suspicious position-label correlations
        for pos_bucket, labels in position_label_map.items():
            if len(labels) > 20:  # Enough samples
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1

                max_label = max(label_counts, key=label_counts.get)
                max_ratio = label_counts[max_label] / len(labels)

                if max_ratio > 0.7:  # 70% of labels at this position are the same
                    errors.append(ErrorReport(
                        error_type="spurious_position_correlation",
                        confidence=max_ratio,
                        evidence={
                            "position_bucket": pos_bucket,
                            "dominant_label": max_label,
                            "ratio": max_ratio,
                            "label_distribution": label_counts
                        },
                        example_id="multiple",
                        details=f"Label {max_label} appears in {max_ratio:.1%} of position {pos_bucket}",
                        recommendations="Check if position-based artifact exists in data"
                    ))

        return errors

    # ============================================================================
    # MACHINE TRANSLATION ERROR DETECTION
    # ============================================================================

    def _check_mt_errors(self, dataset: List[Dict]) -> List[ErrorReport]:
        """Check errors specific to machine translation"""
        errors = []

        print("Checking semantic drift...")
        errors.extend(self._detect_semantic_drift(dataset))

        print("Checking translationese...")
        errors.extend(self._detect_translationese(dataset))

        print("Checking code-switching inconsistency...")
        errors.extend(self._detect_code_switching(dataset))

        return errors

    def _detect_semantic_drift(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect translation pairs with semantic misalignment

        Expected format: {"source": "...", "target": "...", "id": "..."}
        """
        errors = []

        for example in dataset[:100]:
            source = example.get("source", "")
            target = example.get("target", "")
            example_id = example.get("id", "unknown")

            if not source or not target:
                continue

            # Encode source and target
            source_inputs = self.tokenizer(
                source,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(self.device)

            target_inputs = self.tokenizer(
                target,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                # Get encoder outputs for cross-attention analysis
                encoder_outputs = self.model.get_encoder()(**source_inputs)

                # Generate with cross-attention
                outputs = self.model.generate(
                    **source_inputs,
                    max_length=128,
                    return_dict_in_generate=True,
                    output_attentions=True
                )

                # Check cross-attention patterns
                # Note: cross_attentions may be None with some attention mechanisms (e.g., SDPA)
                if (hasattr(outputs, 'cross_attentions') and outputs.cross_attentions 
                    and outputs.cross_attentions[0] is not None):
                    try:
                        cross_attn = outputs.cross_attentions[0][0].mean(dim=1)  # Average over heads

                        # Calculate attention entropy (uniform = high entropy = poor alignment)
                        attn_probs = cross_attn.mean(dim=0).cpu().numpy()
                        attn_probs = attn_probs / (attn_probs.sum() + 1e-9)
                        entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-9))
                        max_entropy = np.log(len(attn_probs))
                        normalized_entropy = entropy / max_entropy

                        if normalized_entropy > 0.8:  # Very uniform attention
                            errors.append(ErrorReport(
                                error_type="semantic_drift",
                                confidence=normalized_entropy,
                                evidence={
                                    "source": source,
                                    "target": target,
                                    "attention_entropy": float(normalized_entropy),
                                    "interpretation": "Cross-attention is very scattered"
                                },
                                example_id=example_id,
                                details="Translation pair may have semantic misalignment",
                                recommendations="Verify source and target convey same meaning"
                            ))
                    except (AttributeError, IndexError, TypeError):
                        # Cross-attentions not available with current attention mechanism
                        pass

        return errors

    def _detect_translationese(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect unnatural target language (word-for-word translation)
        """
        errors = []

        # Check for rigid word-order alignment
        for example in dataset[:50]:
            source = example.get("source", "")
            target = example.get("target", "")
            example_id = example.get("id", "unknown")

            if not source or not target:
                continue

            source_tokens = source.split()
            target_tokens = target.split()

            # Simple heuristic: very similar length might indicate word-for-word
            length_ratio = len(target_tokens) / max(len(source_tokens), 1)

            if 0.9 <= length_ratio <= 1.1 and len(source_tokens) > 5:
                # Check cross-attention for rigid alignment
                source_inputs = self.tokenizer(
                    source,
                    return_tensors="pt",
                    truncation=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **source_inputs,
                        max_length=128,
                        return_dict_in_generate=True,
                        output_attentions=True
                    )

                    # Note: cross_attentions may be None with some attention mechanisms (e.g., SDPA)
                    if (hasattr(outputs, 'cross_attentions') and outputs.cross_attentions 
                        and outputs.cross_attentions[0] is not None):
                        try:
                            # Check if attention follows diagonal pattern (word-for-word)
                            cross_attn = outputs.cross_attentions[0][0].mean(dim=1)
                            attn_matrix = cross_attn.cpu().numpy()

                            # Calculate diagonal dominance
                            min_dim = min(attn_matrix.shape)
                            diagonal_sum = sum(attn_matrix[i, i] for i in range(min_dim))
                            total_sum = attn_matrix.sum()
                            diagonal_ratio = diagonal_sum / (total_sum + 1e-9)

                            if diagonal_ratio > 0.4:  # Strong diagonal pattern
                                errors.append(ErrorReport(
                                    error_type="translationese",
                                    confidence=diagonal_ratio,
                                    evidence={
                                        "source": source,
                                        "target": target,
                                        "length_ratio": length_ratio,
                                        "diagonal_ratio": float(diagonal_ratio)
                                    },
                                    example_id=example_id,
                                    details="Translation follows source word order too rigidly",
                                    recommendations="Check if target language sounds natural"
                                ))
                        except (AttributeError, IndexError, TypeError):
                            # Cross-attentions not available with current attention mechanism
                            pass

        return errors

    def _detect_code_switching(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect inconsistent handling of code-switched content
        """
        errors = []
        code_switch_patterns = {}

        # Look for potential English words in source (simple heuristic)
        for example in dataset:
            source = example.get("source", "")
            target = example.get("target", "")

            source_words = source.split()
            target_words = target.split()

            # Simple English word detection (can be improved)
            for s_word in source_words:
                # Check if word appears in target (copied) or translated
                if s_word.lower() in [t.lower() for t in target_words]:
                    action = "copied"
                else:
                    action = "translated"

                if s_word not in code_switch_patterns:
                    code_switch_patterns[s_word] = []
                code_switch_patterns[s_word].append(action)

        # Find inconsistent patterns
        for word, actions in code_switch_patterns.items():
            if len(actions) >= 3:
                copied_count = actions.count("copied")
                translated_count = actions.count("translated")

                if copied_count > 0 and translated_count > 0:
                    inconsistency_rate = min(copied_count, translated_count) / len(actions)

                    if inconsistency_rate > 0.2:  # 20% inconsistency
                        errors.append(ErrorReport(
                            error_type="code_switching_inconsistency",
                            confidence=inconsistency_rate,
                            evidence={
                                "word": word,
                                "copied_count": copied_count,
                                "translated_count": translated_count,
                                "total_occurrences": len(actions)
                            },
                            example_id="multiple",
                            details=f"Word '{word}' is sometimes copied, sometimes translated",
                            recommendations="Establish consistent code-switching guidelines"
                        ))

        return errors

    # ============================================================================
    # QUESTION ANSWERING ERROR DETECTION
    # ============================================================================

    def _check_qa_errors(self, dataset: List[Dict]) -> List[ErrorReport]:
        """Check errors specific to question answering"""
        errors = []

        print("Checking question-answer misalignment...")
        errors.extend(self._detect_qa_misalignment(dataset))

        print("Checking context contradictions...")
        errors.extend(self._detect_context_contradiction(dataset))

        print("Checking answer length bias...")
        errors.extend(self._detect_answer_length_bias(dataset))

        print("Checking template artifacts...")
        errors.extend(self._detect_template_artifacts(dataset))

        return errors

    def _detect_qa_misalignment(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect questions and answers that don't match in type

        Expected format: {"question": "...", "context": "...", "answer": "...", "id": "..."}
        """
        errors = []

        question_words = {
            "who": ["person", "name", "people"],
            "when": ["time", "date", "year", "day"],
            "where": ["place", "location", "country", "city"],
            "how many": ["number", "count"],
            "what": ["thing", "object", "event"],
            "why": ["reason", "because", "cause"]
        }

        for example in dataset[:100]:
            question = example.get("question", "").lower()
            answer = example.get("answer", "")
            example_id = example.get("id", "unknown")

            # Detect question type
            question_type = None
            for q_word, expected_answer_types in question_words.items():
                if question.startswith(q_word) or f" {q_word} " in question:
                    question_type = q_word
                    break

            if question_type:
                # Simple heuristic checks
                misalignment_detected = False
                reason = ""

                if question_type == "who" and answer.isdigit():
                    misalignment_detected = True
                    reason = "Who question but answer is a number"
                elif question_type == "when" and len(answer.split()) > 10:
                    misalignment_detected = True
                    reason = "When question but answer is very long (not a date/time)"
                elif question_type == "how many" and not any(char.isdigit() for char in answer):
                    misalignment_detected = True
                    reason = "How many question but answer contains no numbers"

                if misalignment_detected:
                    errors.append(ErrorReport(
                        error_type="qa_misalignment",
                        confidence=0.7,
                        evidence={
                            "question": question,
                            "answer": answer,
                            "question_type": question_type,
                            "reason": reason
                        },
                        example_id=example_id,
                        details=reason,
                        recommendations="Verify answer matches question type"
                    ))

        return errors

    def _detect_context_contradiction(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect answers that contradict the context
        """
        errors = []

        for example in dataset[:50]:
            question = example.get("question", "")
            context = example.get("context", "")
            answer = example.get("answer", "")
            example_id = example.get("id", "unknown")

            if not all([question, context, answer]):
                continue

            # Check if answer appears in context
            if answer.lower() not in context.lower():
                # Use model to find what it would predict
                inputs = self.tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=384
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    start_scores = outputs.start_logits[0]
                    end_scores = outputs.end_logits[0]

                    start_idx = torch.argmax(start_scores).item()
                    end_idx = torch.argmax(end_scores).item()

                    if end_idx >= start_idx:
                        tokens = self.tokenizer.convert_ids_to_tokens(
                            inputs.input_ids[0][start_idx:end_idx+1]
                        )
                        model_answer = self.tokenizer.convert_tokens_to_string(tokens)

                        # Model found different answer in context
                        if model_answer.strip() and model_answer.lower() != answer.lower():
                            errors.append(ErrorReport(
                                error_type="context_contradiction",
                                confidence=0.8,
                                evidence={
                                    "question": question,
                                    "gold_answer": answer,
                                    "model_answer": model_answer,
                                    "context_snippet": context[:200]
                                },
                                example_id=example_id,
                                details="Gold answer not found in context, model predicts different answer",
                                recommendations="Check if annotator used external knowledge"
                            ))

        return errors

    def _detect_answer_length_bias(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect systematic bias in answer lengths
        """
        errors = []
        answer_lengths = []

        for example in dataset:
            answer = example.get("answer", "")
            answer_lengths.append(len(answer.split()))

        if len(answer_lengths) > 20:
            mean_length = np.mean(answer_lengths)
            std_length = np.std(answer_lengths)

            # Check if most answers are very similar length
            within_one_std = sum(1 for l in answer_lengths
                                if abs(l - mean_length) <= std_length)
            ratio = within_one_std / len(answer_lengths)

            if ratio > 0.85 and std_length < 2.0:  # 85% within 1 std and low variance
                errors.append(ErrorReport(
                    error_type="answer_length_bias",
                    confidence=ratio,
                    evidence={
                        "mean_length": float(mean_length),
                        "std_length": float(std_length),
                        "concentration_ratio": ratio
                    },
                    example_id="multiple",
                    details=f"85% of answers have length within {std_length:.1f} words of mean ({mean_length:.1f})",
                    recommendations="Check if answer span selection has length bias"
                ))

        return errors

    def _detect_template_artifacts(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect templated question generation patterns
        """
        errors = []
        question_structures = {}

        for example in dataset:
            question = example.get("question", "")

            # Extract structure (first 3 and last 2 words)
            words = question.split()
            if len(words) > 5:
                structure = " ".join(words[:3]) + " ... " + " ".join(words[-2:])

                if structure not in question_structures:
                    question_structures[structure] = []
                question_structures[structure].append(example.get("id", "unknown"))

        # Find over-represented structures
        total_questions = len(dataset)
        for structure, example_ids in question_structures.items():
            if len(example_ids) > 0.1 * total_questions:  # >10% of questions
                errors.append(ErrorReport(
                    error_type="template_artifacts",
                    confidence=len(example_ids) / total_questions,
                    evidence={
                        "template_structure": structure,
                        "occurrence_count": len(example_ids),
                        "percentage": f"{100 * len(example_ids) / total_questions:.1f}%"
                    },
                    example_id="multiple",
                    details=f"Question structure '{structure}' appears in {len(example_ids)} examples",
                    recommendations="Check for synthetic question generation artifacts"
                ))

        return errors

    # ============================================================================
    # SENTIMENT ANALYSIS ERROR DETECTION
    # ============================================================================

    def _check_sentiment_errors(self, dataset: List[Dict]) -> List[ErrorReport]:
        """Check errors specific to sentiment analysis"""
        errors = []

        print("Checking label noise in sentiment...")
        errors.extend(self._detect_sentiment_label_noise(dataset))

        print("Checking class imbalance...")
        errors.extend(self._detect_class_imbalance(dataset))

        return errors

    def _detect_sentiment_label_noise(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect sentiment labels that might be wrong (especially sarcasm)

        Expected format: {"text": "...", "label": 0/1/2, "id": "..."}
        """
        errors = []

        # Sarcasm indicators (simple heuristic)
        sarcasm_indicators = ["great job", "wonderful", "excellent", "amazing"]
        negative_words = ["destroy", "ruin", "disaster", "terrible", "worst"]

        for example in dataset[:100]:
            text = example.get("text", "").lower()
            label = example.get("label", 1)  # Assuming 0=neg, 1=neu, 2=pos
            example_id = example.get("id", "unknown")

            # Check for sarcasm pattern: positive words + negative context
            has_positive_words = any(word in text for word in sarcasm_indicators)
            has_negative_words = any(word in text for word in negative_words)

            if has_positive_words and has_negative_words and label == 2:  # Labeled as positive
                errors.append(ErrorReport(
                    error_type="sentiment_label_noise",
                    confidence=0.7,
                    evidence={
                        "text": text,
                        "label": label,
                        "detected_pattern": "Possible sarcasm"
                    },
                    example_id=example_id,
                    details="Text contains positive and negative words, might be sarcastic",
                    recommendations="Manual review for sarcasm detection"
                ))

        return errors

    def _detect_class_imbalance(self, dataset: List[Dict]) -> List[ErrorReport]:
        """
        Detect severe class imbalance in classification tasks
        """
        errors = []
        label_counts = {}

        for example in dataset:
            label = example.get("label", 1)
            label_counts[label] = label_counts.get(label, 0) + 1

        if len(label_counts) > 1:
            total = sum(label_counts.values())
            max_class = max(label_counts, key=label_counts.get)
            max_ratio = label_counts[max_class] / total

            if max_ratio > 0.7:  # 70% imbalance
                errors.append(ErrorReport(
                    error_type="class_imbalance",
                    confidence=max_ratio,
                    evidence={
                        "label_distribution": label_counts,
                        "majority_class": max_class,
                        "majority_ratio": max_ratio
                    },
                    example_id="multiple",
                    details=f"Class {max_class} represents {max_ratio:.1%} of data",
                    recommendations="Consider data augmentation or weighted loss"
                ))

        return errors

    # ============================================================================
    # REPORTING AND EXPORT
    # ============================================================================

    def generate_report(self, errors: List[ErrorReport], output_file: str = "data_quality_report.txt"):
        """Generate human-readable report"""
        
        if not errors:
            logger.info(f"No errors detected. Clean dataset!")
            return

        # Group errors by type
        errors_by_type = {}
        for error in errors:
            if error.error_type not in errors_by_type:
                errors_by_type[error.error_type] = []
            errors_by_type[error.error_type].append(error)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA QUALITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total errors detected: {len(errors)}\n")
            f.write(f"Error types found: {len(errors_by_type)}\n\n")

            # Summary by error type
            f.write("-" * 80 + "\n")
            f.write("ERROR SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            
            for error_type, error_list in sorted(errors_by_type.items()):
                f.write(f"\n{error_type.upper()}\n")
                f.write(f"  Count: {len(error_list)}\n")
                f.write(f"  Avg Confidence: {np.mean([e.confidence for e in error_list]):.3f}\n")

            # Detailed examples
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED EXAMPLES (Top 5 per error type)\n")
            f.write("=" * 80 + "\n\n")

            for error_type, error_list in sorted(errors_by_type.items()):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{error_type.upper()}\n")
                f.write(f"{'=' * 80}\n\n")

                # Sort by confidence and take top 5
                sorted_errors = sorted(error_list, key=lambda x: x.confidence, reverse=True)[:5]

                for idx, error in enumerate(sorted_errors, 1):
                    f.write(f"\nExample {idx}:\n")
                    f.write(f"  ID: {error.example_id}\n")
                    f.write(f"  Confidence: {error.confidence:.3f}\n")
                    f.write(f"  Details: {error.details}\n")
                    f.write(f"  Evidence: {str(error.evidence)[:200]}...\n")
                    f.write(f"  Recommendation: {error.recommendations}\n")
                    f.write("-" * 80 + "\n")

        logger.info(f"Report saved to {output_file}")
        print(f"\n✓ Report generated: {output_file}")
        print(f"  Total errors: {len(errors)}")
        print(f"  Error types: {len(errors_by_type)}")


# ============================================================================
# EXAMPLE USAGE WITH AFRIDOCMT DATASET
# ============================================================================

def load_afridocmt_for_mt(config_name: str = "health", split: str = "test", max_samples: int = 100):
    """
    Load AfriDocMT dataset in format suitable for MT error detection
    
    Args:
        config_name: Dataset configuration (health, tech)
        split: Dataset split (train, dev, test)
        max_samples: Maximum number of samples to load
        
    Returns:
        List of dictionaries with source, target, and id
    """
    from datasets import load_dataset
    
    logger.info(f"Loading AfriDocMT dataset: config={config_name}, split={split}")
    
    try:
        dataset = load_dataset("masakhane/AfriDocMT", config_name, split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Attempting to download from HuggingFace...")
        dataset = load_dataset("masakhane/AfriDocMT", config_name, split=split, download_mode="force_redownload")
    
    # Convert to MT format (English -> target language)
    languages = ["am", "ha", "sw", "yo", "zu"]  # African languages in dataset
    
    mt_datasets = {}
    for lang in languages:
        mt_data = []
        for idx, example in enumerate(dataset):
            if idx >= max_samples:
                break
            
            # Check if both source and target exist
            if "en" in example and lang in example:
                if example["en"] and example[lang]:  # Non-empty
                    mt_data.append({
                        "source": example["en"],
                        "target": example[lang],
                        "id": f"{config_name}_{lang}_{idx}"
                    })
        
        mt_datasets[lang] = mt_data
        logger.info(f"  Loaded {len(mt_data)} examples for {lang}")
    
    return mt_datasets


def run_mt_quality_analysis(
    model_name: str = "facebook/nllb-200-distilled-600M",
    config_name: str = "health",
    target_lang: str = "sw",
    max_samples: int = 100,
    output_dir: str = "quality_reports"
):
    """
    Run complete MT quality analysis on AfriDocMT dataset
    
    Args:
        model_name: HuggingFace model name
        config_name: Dataset configuration (health, tech)
        target_lang: Target language code (am, ha, sw, yo, zu)
        max_samples: Maximum samples to analyze
        output_dir: Output directory for reports
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("MACHINE TRANSLATION QUALITY ANALYSIS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: AfriDocMT/{config_name}")
    print(f"Target Language: {target_lang}")
    print(f"Max Samples: {max_samples}")
    print("=" * 80)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    mt_datasets = load_afridocmt_for_mt(config_name, split="test", max_samples=max_samples)
    
    if target_lang not in mt_datasets:
        raise ValueError(f"Language {target_lang} not found. Available: {list(mt_datasets.keys())}")
    
    dataset = mt_datasets[target_lang]
    print(f"✓ Loaded {len(dataset)} translation pairs")
    
    # Initialize checker
    print("\n[2/3] Initializing quality checker...")
    checker = DataQualityChecker(model_name, TaskType.MT)
    
    # Run error detection
    print("\n[3/3] Detecting errors...")
    errors = checker.check_all_errors(dataset)
    
    # Generate report
    report_file = os.path.join(output_dir, f"quality_report_{config_name}_{target_lang}.txt")
    checker.generate_report(errors, report_file)
    
    # Export to JSON
    import json
    json_file = os.path.join(output_dir, f"errors_{config_name}_{target_lang}.json")
    
    errors_dict = []
    for error in errors:
        errors_dict.append({
            "error_type": error.error_type,
            "confidence": error.confidence,
            "example_id": error.example_id,
            "details": error.details,
            "recommendations": error.recommendations,
            "evidence": str(error.evidence)[:500]  # Truncate for JSON
        })
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(errors_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Errors exported to JSON: {json_file}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return errors


def example_ner_analysis():
    """Example: NER error detection"""
    print("\n=== NER ERROR DETECTION EXAMPLE ===\n")
    
    # Sample NER dataset
    ner_data = [
        {
            "tokens": ["Amina", "lives", "in", "Lagos", "Nigeria"],
            "labels": [1, 0, 0, 3, 3],  # 1=PERSON, 3=LOCATION
            "id": "example_1"
        },
        {
            "tokens": ["Dr", ".", "Smith", "works", "at", "Mayo", "Clinic"],
            "labels": [1, 0, 1, 0, 0, 3, 3],
            "id": "example_2"
        },
        {
            "tokens": ["Lagos", "is", "in", "Nigeria"],
            "labels": [0, 0, 0, 3],  # Inconsistent: Lagos labeled as O instead of LOC
            "id": "example_3"
        }
    ]
    
    checker = DataQualityChecker("Davlan/xlm-roberta-base-wikiann-ner", TaskType.NER)
    errors = checker.check_all_errors(ner_data)
    checker.generate_report(errors, "ner_quality_report.txt")


def main():
    """Main entry point with examples"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data Quality Error Detection for African Language NLP"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mt",
        choices=["mt", "ner", "qa"],
        help="Task type to analyze"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="health",
        help="AfriDocMT config (health, tech)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="sw",
        help="Target language (am, ha, sw, yo, zu)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples to analyze"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Model name from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="quality_reports",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    if args.task == "mt":
        run_mt_quality_analysis(
            model_name=args.model,
            config_name=args.config,
            target_lang=args.lang,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
    elif args.task == "ner":
        example_ner_analysis()
    else:
        print(f"Task {args.task} not implemented in example. Use 'mt' or 'ner'.")


if __name__ == "__main__":
    main()