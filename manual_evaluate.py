#!/usr/bin/env python3
"""
手动评估工具：输入标准答案 → 自动预测 → 计算准确率指标
"""

import argparse
import json
import os
import sys
import base64
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append("./")
from utils.constants import PROMPT_TEMPLATES
import re

load_dotenv(override=True)

# 选6种不同灾害类型，各取第1张
SAMPLE_EVENTS = [
    "guatemala-volcano",  # 火山
    "joplin-tornado",     # 龙卷风
    "hurricane-harvey",   # 洪水/飓风
    "santa-rosa-wildfire", # 野火
    "palu-tsunami",       # 海啸
    "mexico-earthquake",  # 地震
]

DATA_DIR = "RSCC-Data/xbd_subset"

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def show_image(path):
    """用 feh 显示图片"""
    try:
        subprocess.run(["feh", path], check=True)
    except:
        print(f"  (图片位置: {path})")

def get_image_pairs():
    """获取每个灾害类型的第1张图（part1）"""
    pairs = []
    for event in SAMPLE_EVENTS:
        pre_path = os.path.join(DATA_DIR, event, "images", f"{event}_00000000_pre_disaster_part1.png")
        post_path = os.path.join(DATA_DIR, event, "images", f"{event}_00000000_post_disaster_part1.png")
        if os.path.exists(pre_path) and os.path.exists(post_path):
            pairs.append({
                "event": event,
                "pre_image": pre_path,
                "post_image": post_path,
            })
    return pairs

def get_prediction(client, model, pre_path, post_path, prompt_text):
    """调用 API 获取预测"""
    pre_b64 = encode_image(pre_path)
    post_b64 = encode_image(post_path)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pre_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{post_b64}"}},
            {"type": "text", "text": prompt_text},
        ]
    }]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.01,
    )
    return response.choices[0].message.content

def calculate_metrics(predictions, references):
    """计算评估指标"""
    from bert_score import score as bert_score
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    print("\n📊 正在计算指标...")
    
    # 1. BERTScore
    print("  ⏳ 计算 BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    bert_f1 = F1.mean().item()
    
    # 2. 余弦相似度（sentence-transformers）
    print("  ⏳ 计算语义相似度...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pred_embeds = model.encode(predictions, normalize_embeddings=True)
    ref_embeds = model.encode(references, normalize_embeddings=True)
    similarities = cosine_similarity(pred_embeds, ref_embeds).diagonal()
    avg_similarity = similarities.mean()
    
    # 3. 平均描述长度
    avg_len = np.mean([len(c.split()) for c in predictions])
    
    return {
        "bertscore_f1": float(bert_f1),
        "cosine_similarity": float(avg_similarity),
        "avg_caption_length": float(avg_len),
        "num_samples": len(predictions),
    }

def main():
    parser = argparse.ArgumentParser(description="手动评估灾害描述准确率")
    parser.add_argument("--output", default="./output/manual_eval_result.json", help="结果输出路径")
    parser.add_argument("--prompt_type", default="naive", choices=["naive", "textual"], help="使用的提示模板")
    args = parser.parse_args()
    
    # 初始化 API
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("API_BASE_URL")
    model = os.getenv("QVQ_MODEL_NAME", "qwen3-vl-plus")
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 获取样本图片对
    pairs = get_image_pairs()
    
    if args.prompt_type == "naive":
        prompt_text = PROMPT_TEMPLATES["naive"].strip()
    else:
        prompt_text = PROMPT_TEMPLATES["textual"].format(
            disaster_type="{disaster_type}",
            number={"all": "{total}", "no-damage": "{no}", "minor-damage": "{minor}", 
                    "major-damage": "{major}", "destroyed": "{destroyed}", "un-classified": "{unknown}"}
        )
        # 简化 textual prompt，去掉占位符
        prompt_text = "Describe the changes between these two satellite images before and after a natural disaster."
    
    print(f"\n{'='*60}")
    print(f"🌋 灾害变化描述 - 手动评估工具")
    print(f"🤖 模型: {model}")
    print(f"📝 提示模板: {args.prompt_type}")
    print(f"{'='*60}\n")
    
    predictions = []
    references = []
    results = []
    
    for i, pair in enumerate(pairs):
        event = pair["event"]
        print(f"\n--- 样本 {i+1}/{len(pairs)}: {event} ---")
        print(f"\n📸 灾害前图片:")
        show_image(pair["pre_image"])
        input("  按 Enter 键继续查看灾害后图片...")
        
        print(f"\n📸 灾害后图片:")
        show_image(pair["post_image"])
        
        print(f"\n✏️  请根据你看到的图片，输入一段标准答案描述（英文）：")
        print(f"  提示：描述灾害前后的变化、建筑物损毁情况等")
        gt = input("  ➤ ")
        while not gt.strip():
            print("  描述不能为空，请重新输入：")
            gt = input("  ➤ ")
        
        references.append(gt.strip())
        
        print(f"\n⏳ 正在调用 {model} 生成预测...")
        try:
            pred = get_prediction(client, model, pair["pre_image"], pair["post_image"], prompt_text)
            predictions.append(pred)
            print(f"  ✅ 预测结果: {pred[:150]}...")
        except Exception as e:
            print(f"  ❌ API调用失败: {e}")
            predictions.append("")
        
        results.append({
            "event": event,
            "pre_image": pair["pre_image"],
            "post_image": pair["post_image"],
            "ground_truth": gt.strip(),
            "prediction": predictions[-1],
        })
        
        print(f"\n--- 样本 {i+1} 完成 ---")
        if i < len(pairs) - 1:
            input("\n按 Enter 继续下一个样本...")
    
    # 计算指标
    print(f"\n{'='*60}")
    print("📊 开始计算评估指标...")
    
    metrics = calculate_metrics(predictions, references)
    
    # 输出结果
    print(f"\n{'='*60}")
    print("✅ 评估完成！结果如下：")
    print(f"{'='*60}")
    print(f"📈 BERTScore F1:     {metrics['bertscore_f1']:.4f}")
    print(f"📈 语义相似度:       {metrics['cosine_similarity']:.4f}")
    print(f"📈 平均描述长度:     {metrics['avg_caption_length']:.1f} 词")
    print(f"📊 评估样本数:       {metrics['num_samples']} 组")
    
    # 保存结果
    output = {
        "model": model,
        "prompt_type": args.prompt_type,
        "metrics": metrics,
        "samples": results,
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 详细结果已保存到: {args.output}")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
