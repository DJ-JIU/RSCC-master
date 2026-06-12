"""
简化版评估脚本 - 使用论文标准的 sentence-t5-xxl 和 roberta-large
sentence-t5-xxl 太大（~11B）放不进 GPU，使用 CPU 推理
"""
import json
import logging
import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 设置代理
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_jsonl(file_path):
    with open(file_path) as f:
        return [json.loads(line) for line in f]


def extract_filename(path):
    return os.path.basename(path)


def calculate_metrics(source_texts, ground_truths, sentence_model, roberta_model, roberta_tokenizer):
    """计算所有评估指标"""
    
    # 1. ST5 语义相似度 (论文标准)
    source_embeds = sentence_model.encode(source_texts, normalize_embeddings=True, show_progress_bar=True)
    gt_embeds = sentence_model.encode(ground_truths, normalize_embeddings=True, show_progress_bar=True)
    similarities = cosine_similarity(source_embeds, gt_embeds).diagonal()
    
    # 2. 描述长度
    caption_lengths = [len(caption.split()) for caption in source_texts]
    gt_lengths = [len(caption.split()) for caption in ground_truths]
    
    # 3. ROUGE-L (LCS)
    rouge_scores = []
    for src, ref in zip(source_texts, ground_truths):
        src_words = src.lower().split()
        ref_words = ref.lower().split()
        lcs_len = longest_common_subsequence(src_words, ref_words)
        recall = lcs_len / len(ref_words) if len(ref_words) > 0 else 0
        precision = lcs_len / len(src_words) if len(src_words) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        rouge_scores.append(f1)
    
    # 4. METEOR (unigram 匹配)
    meteor_scores = []
    for src, ref in zip(source_texts, ground_truths):
        src_words = set(src.lower().split())
        ref_words = set(ref.lower().split())
        recall = len(src_words & ref_words) / len(ref_words) if len(ref_words) > 0 else 0
        precision = len(src_words & ref_words) / len(src_words) if len(src_words) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        meteor_scores.append(f1)
    
    # 5. BLEU-1
    bleu1_scores = []
    for src, ref in zip(source_texts, ground_truths):
        src_words = src.lower().split()
        ref_words = ref.lower().split()
        ref_counts = defaultdict(int)
        for w in ref_words:
            ref_counts[w] += 1
        matches = 0
        for w in src_words:
            if ref_counts[w] > 0:
                matches += 1
                ref_counts[w] -= 1
        bleu1_scores.append(matches / len(src_words) if len(src_words) > 0 else 0)
    
    # 6. BERTScore (用 RoBERTa-large)
    with torch.no_grad():
        cand_inputs = roberta_tokenizer(source_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        ref_inputs = roberta_tokenizer(ground_truths, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        cand_embeddings = roberta_model(**cand_inputs).last_hidden_state.mean(dim=1)
        ref_embeddings = roberta_model(**ref_inputs).last_hidden_state.mean(dim=1)
        
        bert_scores = torch.nn.functional.cosine_similarity(cand_embeddings, ref_embeddings).cpu().numpy()
    
    return {
        "st5_similarity": float(np.mean(similarities)),
        "st5_similarity_std": float(np.std(similarities)),
        "bertscore": float(np.mean(bert_scores)),
        "rouge_l": float(np.mean(rouge_scores)),
        "meteor": float(np.mean(meteor_scores)),
        "bleu_1": float(np.mean(bleu1_scores)),
        "avg_caption_length": float(np.mean(caption_lengths)),
        "avg_gt_length": float(np.mean(gt_lengths)),
    }


def longest_common_subsequence(a, b):
    """计算 LCS 长度"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def evaluate_models(gt_path, pred_path, sentence_model, roberta_model, roberta_tokenizer):
    """评估所有模型结果"""
    
    logger.info(f"加载 ground truth: {gt_path}")
    gt_data = load_jsonl(gt_path)
    logger.info(f"  共 {len(gt_data)} 条")
    
    logger.info(f"加载 predictions: {pred_path}")
    pred_data = load_jsonl(pred_path)
    logger.info(f"  共 {len(pred_data)} 条")
    
    # 按文件名匹配
    gt_mapping = {}
    for item in gt_data:
        key = (extract_filename(item["pre_image"]), extract_filename(item["post_image"]))
        gt_mapping[key] = item["change_caption"]
    
    # 按 model_id + inference_type 分组
    model_results = defaultdict(lambda: {"source_texts": [], "ground_truths": []})
    matched = 0
    
    for pred in pred_data:
        if "pre_image" not in pred:
            # 可能是 Qwen-VL 格式（有 images 字段）
            continue
        key = (extract_filename(pred["pre_image"]), extract_filename(pred["post_image"]))
        
        if key in gt_mapping:
            matched += 1
            model_key = f"{pred['model_id']} ({pred['inference_type']})"
            model_results[model_key]["source_texts"].append(pred["change_caption"])
            model_results[model_key]["ground_truths"].append(gt_mapping[key])
    
    logger.info(f"  匹配到 {matched} 对")
    
    # 计算指标
    results = {}
    for model_key, data in sorted(model_results.items()):
        logger.info(f"  计算: {model_key}")
        results[model_key] = calculate_metrics(
            data["source_texts"], data["ground_truths"], sentence_model, roberta_model, roberta_tokenizer
        )
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", default="/home/long/RSCC-master/output/full_qvq.jsonl")
    parser.add_argument("--predictions_file", default="/home/long/RSCC-master/output/test.jsonl")
    parser.add_argument("--output_file", default="/home/long/RSCC-master/output/evaluation_results/metrics_results.json")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 加载 sentence model（使用论文标准的 sentence-t5-xxl，用 CPU 避免 OOM）
    logger.info("加载 sentence-t5-xxl 模型（CPU）...")
    sentence_model = SentenceTransformer(
        '/home/long/RSCC-master/models/sentence-transformers/sentence-t5-xxl',
        device='cpu',
    )
    logger.info("✅ sentence-t5-xxl 加载完成")
    
    # 加载 RoBERTa 用于 BERTScore（用 CPU 避免 OOM）
    logger.info("加载 RoBERTa-large 模型（CPU）...")
    from transformers import AutoTokenizer, AutoModel
    roberta_tokenizer = AutoTokenizer.from_pretrained(
        '/home/long/RSCC-master/models/FacebookAI/roberta-large',
        local_files_only=True,
    )
    roberta_model = AutoModel.from_pretrained(
        '/home/long/RSCC-master/models/FacebookAI/roberta-large',
        local_files_only=True,
    ).to('cpu')
    logger.info("✅ RoBERTa-large 加载完成")
    
    # 评估
    results = evaluate_models(args.ground_truth_file, args.predictions_file, sentence_model, roberta_model, roberta_tokenizer)
    
    # 输出
    logger.info("\n" + "="*80)
    logger.info("评估结果")
    logger.info("="*80)
    for model_key, metrics in results.items():
        logger.info(f"\n{model_key}:")
        logger.info(f"  BLEU-1:     {metrics['bleu_1']:.2%}")
        logger.info(f"  ROUGE-L:    {metrics['rouge_l']:.2%}")
        logger.info(f"  METEOR:     {metrics['meteor']:.2%}")
        logger.info(f"  语义相似度: {metrics['semantic_similarity']:.2%}")
        logger.info(f"  平均描述长度: {metrics['avg_caption_length']:.0f} (GT: {metrics['avg_gt_length']:.0f})")
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
