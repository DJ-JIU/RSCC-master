#!/usr/bin/env python3
"""
自评估脚本：对 qwen3-vl-plus 的预测结果进行质量分析
不需要对比文件，直接统计生成结果的质量特征
"""

import json
import sys
import os
import numpy as np
from collections import Counter, defaultdict
import argparse

sys.path.append("./")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("./")


def load_jsonl(file_path):
    with open(file_path) as f:
        return [json.loads(line) for line in f]


def extract_event_name(path):
    """从路径中提取灾害事件名称"""
    parts = path.split("/")
    for part in parts:
        if part in [
            "guatemala-volcano", "lower-puna-volcano",
            "pinery-bushfire", "portugal-wildfire", "santa-rosa-wildfire",
            "socal-fire", "woolsey-fire",
            "hurricane-florence", "joplin-tornado", "moore-tornado",
            "hurricane-matthew", "tuscaloosa-tornado",
            "mexico-earthquake",
            "hurricane-michael", "hurricane-harvey",
            "midwest-flooding", "nepal-flooding",
            "palu-tsunami", "sunda-tsunami",
        ]:
            return part
    return "unknown"


def classify_disaster_type(event_name):
    disaster_map = {
        "volcano": ["guatemala-volcano", "lower-puna-volcano"],
        "wildfire": ["pinery-bushfire", "portugal-wildfire", "santa-rosa-wildfire",
                     "socal-fire", "woolsey-fire"],
        "storm": ["hurricane-florence", "joplin-tornado", "moore-tornado",
                  "hurricane-matthew", "tuscaloosa-tornado"],
        "earthquake": ["mexico-earthquake"],
        "flooding": ["hurricane-michael", "hurricane-harvey",
                     "midwest-flooding", "nepal-flooding"],
        "tsunami": ["palu-tsunami", "sunda-tsunami"],
    }
    for category, names in disaster_map.items():
        if event_name in names:
            return category
    return "unknown"


def analyze_caption_quality(captions):
    """分析预测描述的质量特征"""
    stats = {}
    
    # 描述长度
    lengths = [len(c.split()) for c in captions]
    stats["length"] = {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
    }
    
    # 关键词覆盖
    disaster_keywords = {
        "damage": ["damage", "destroy", "collapse", "debris", "rubble"],
        "fire": ["fire", "burn", "wildfire", "scorch", "ash", "smoke"],
        "water": ["flood", "water", "inundat", "submerg", "saturat"],
        "wind": ["wind", "tornado", "storm", "hurricane"],
        "earthquake": ["earthquake", "seismic", "fault", "crack"],
        "volcano": ["volcano", "eruption", "lava", "pyroclastic", "ash"],
        "building": ["building", "structure", "roof", "wall", "residential"],
        "level": ["level 0", "level 1", "level 2", "level 3", "no damage",
                  "minor damage", "major damage", "destroyed"],
    }
    
    keyword_stats = {}
    for category, keywords in disaster_keywords.items():
        count = sum(1 for c in captions if any(k in c.lower() for k in keywords))
        keyword_stats[category] = {
            "count": count,
            "percentage": round(count / len(captions) * 100, 1)
        }
    stats["keywords"] = keyword_stats
    
    # 句子数量
    sentences = [c.count(".") + c.count("!") + c.count("?") for c in captions]
    stats["sentences"] = {
        "min": int(np.min(sentences)),
        "max": int(np.max(sentences)),
        "mean": float(np.mean(sentences)),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="自评估 qwen3-vl-plus 预测结果")
    parser.add_argument("--input_file", default="./output/full_qvq.jsonl")
    parser.add_argument("--output_file", default="./output/self_eval_result.json")
    args = parser.parse_args()
    
    print(f"📂 加载预测结果: {args.input_file}")
    data = load_jsonl(args.input_file)
    print(f"📊 共 {len(data)} 条预测\n")
    
    # 按灾害类型分组
    by_event = defaultdict(list)
    by_disaster = defaultdict(list)
    
    for item in data:
        event = extract_event_name(item["pre_image"])
        disaster_type = classify_disaster_type(event)
        by_event[event].append(item)
        by_disaster[disaster_type].append(item)
    
    # 提取所有描述
    all_captions = [item["change_caption"] for item in data]
    
    # 分析整体质量
    print("=" * 60)
    print("📊 整体质量分析")
    print("=" * 60)
    
    overall_stats = analyze_caption_quality(all_captions)
    
    print(f"\n📏 描述长度统计:")
    print(f"  最短: {overall_stats['length']['min']} 词")
    print(f"  最长: {overall_stats['length']['max']} 词")
    print(f"  平均: {overall_stats['length']['mean']:.1f} 词")
    print(f"  中位数: {overall_stats['length']['median']:.1f} 词")
    
    print(f"\n📝 句子数量:")
    print(f"  平均: {overall_stats['sentences']['mean']:.1f} 句")
    
    print(f"\n🔑 关键词覆盖率:")
    for category, info in overall_stats["keywords"].items():
        bar = "█" * int(info["percentage"] // 5) + "░" * int(20 - info["percentage"] // 5)
        print(f"  {category:12s} {bar} {info['percentage']:5.1f}% ({info['count']}/{len(data)})")
    
    # 按灾害类型分析
    print(f"\n{'=' * 60}")
    print("📊 按灾害类型分析")
    print("=" * 60)
    
    disaster_stats = {}
    for disaster_type, items in sorted(by_disaster.items()):
        captions = [item["change_caption"] for item in items]
        stats = analyze_caption_quality(captions)
        disaster_stats[disaster_type] = stats
        
        print(f"\n{disaster_type.upper()} ({len(items)} 张):")
        print(f"  平均长度: {stats['length']['mean']:.1f} 词")
        print(f"  平均句数: {stats['sentences']['mean']:.1f} 句")
    
    # 语义相似度（自对比）
    print(f"\n{'=' * 60}")
    print("📊 语义多样性分析")
    print("=" * 60)
    
    print("  ⏳ 计算语义嵌入...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_captions, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)
    
    # 平均相似度（越高说明输出越单一）
    avg_sim = (np.sum(sim_matrix) - len(sim_matrix)) / (len(sim_matrix) * (len(sim_matrix) - 1))
    print(f"  平均内部相似度: {avg_sim:.4f}")
    if avg_sim < 0.3:
        print("  ✅ 输出多样性良好")
    elif avg_sim < 0.5:
        print("  ⚠️ 输出有一定重复性")
    else:
        print("  ❌ 输出过于相似，多样性不足")
    
    # 保存结果
    output = {
        "total_samples": len(data),
        "by_disaster_type": {k: len(v) for k, v in sorted(by_disaster.items())},
        "by_event": {k: len(v) for k, v in sorted(by_event.items())},
        "overall_stats": overall_stats,
        "disaster_stats": disaster_stats,
        "diversity_analysis": {
            "avg_internal_similarity": float(avg_sim),
        },
    }
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 评估结果已保存到: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
