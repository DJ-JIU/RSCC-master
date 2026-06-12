"""
AutoEval: 使用 Qwen3-VL-Plus 作为裁判，比较三种 prompt 策略 (zero-shot, textual, visual) 的胜负
"""
import argparse, json, os, sys, random, base64, threading
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import EVALUATION_PROMPT_TEMPLATE

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
)
MODEL_NAME = os.getenv("QVQ_MODEL_NAME", "qwen3-vl-plus")

STRATEGY_NAMES = {
    "zero-shot": "Zero-shot",
    "textual_prompt": "Textual Prompt",
    "visual_prompt": "Visual Prompt",
}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./output/test.jsonl")
    parser.add_argument("--output_dir", default="./output/evaluation_results/autoeval")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="每种比较的样本数（每个比较对比 2 个策略）")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    data = load_jsonl(args.input)
    
    # 按图片对和策略分组
    by_pair = defaultdict(dict)
    for d in data:
        if "pre_image" not in d or "change_caption" not in d:
            continue
        key = (d["pre_image"], d["post_image"])
        by_pair[key][d["inference_type"]] = d["change_caption"]

    print(f"图片对: {len(by_pair)}")

    # 两两比较: (strategy_a, strategy_b)
    comparisons = [
        ("zero-shot", "textual_prompt"),
        ("zero-shot", "visual_prompt"),
        ("textual_prompt", "visual_prompt"),
    ]

    # 验证所有图片对都有三种策略
    valid_pairs = [(k, v) for k, v in by_pair.items()
                   if all(t in v for t in ["zero-shot", "textual_prompt", "visual_prompt"])]
    print(f"三策略齐全的图片对: {len(valid_pairs)}")

    os.makedirs(args.output_dir, exist_ok=True)
    file_lock = threading.Lock()

    for strat_a, strat_b in comparisons:
        name_a = STRATEGY_NAMES[strat_a]
        name_b = STRATEGY_NAMES[strat_b]
        output_path = os.path.join(args.output_dir, f"results_{strat_a}_vs_{strat_b}.jsonl")

        # 已处理
        processed = set()
        if os.path.exists(output_path):
            for line in open(output_path):
                try:
                    e = json.loads(line)
                    processed.add((e["pre_image"], e["post_image"]))
                except:
                    pass

        # 采样未处理的
        candidates = [(k, v) for k, v in valid_pairs if k not in processed]
        random.shuffle(candidates)
        selected = candidates[:args.max_samples]

        print(f"\n{'='*50}")
        print(f"比较: {name_a} vs {name_b}")
        print(f"待处理: {len(selected)}")

        def evaluate_pair(pair_key, pair_values):
            pre_path, post_path = pair_key
            caption_a = pair_values[strat_a]
            caption_b = pair_values[strat_b]

            captions = [
                f"Model_A ({name_a}): {caption_a}",
                f"Model_B ({name_b}): {caption_b}",
            ]
            prompt_text = EVALUATION_PROMPT_TEMPLATE.format(captions="\n".join(captions))

            try:
                pre_img_b64 = encode_image(pre_path)
                post_img_b64 = encode_image(post_path)

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pre_img_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{post_img_b64}"}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }],
                    response_format={"type": "json_object"},
                    temperature=0.01,
                    max_tokens=256,
                )

                result = json.loads(response.choices[0].message.content)
                
                entry = {
                    "pre_image": pre_path,
                    "post_image": post_path,
                    "strategy_a": strat_a,
                    "strategy_b": strat_b,
                    "best_model": result.get("best_model_id", "unknown"),
                    "reason": result.get("reason", ""),
                }
                return entry
            except Exception as e:
                print(f"  错误: {e}")
                return None

        with tqdm(total=len(selected), desc=f"{name_a} vs {name_b}") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(evaluate_pair, k, v): k for k, v in selected}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        with file_lock:
                            with open(output_path, "a") as f:
                                f.write(json.dumps(result) + "\n")
                                f.flush()
                    pbar.update(1)

        # 统计结果
        wins_a, wins_b, ties = 0, 0, 0
        for line in open(output_path):
            e = json.loads(line)
            if e["best_model"] == "Model_A":
                wins_a += 1
            elif e["best_model"] == "Model_B":
                wins_b += 1
            else:
                ties += 1
        total = wins_a + wins_b + ties
        print(f"\n结果: {name_a} wins: {wins_a}/{total} ({wins_a/total*100:.1f}%)")
        print(f"     {name_b} wins: {wins_b}/{total} ({wins_b/total*100:.1f}%)")
        print(f"     平局: {ties}/{total} ({ties/total*100:.1f}%)")

    # 汇总
    print(f"\n{'='*50}")
    print("汇总胜负率:")
    for strat_a, strat_b in comparisons:
        path = os.path.join(args.output_dir, f"results_{strat_a}_vs_{strat_b}.jsonl")
        wins_a, wins_b = 0, 0
        for line in open(path):
            e = json.loads(line)
            if e["best_model"] == "Model_A":
                wins_a += 1
            elif e["best_model"] == "Model_B":
                wins_b += 1
        total = wins_a + wins_b
        if total > 0:
            name_a = STRATEGY_NAMES[strat_a]
            name_b = STRATEGY_NAMES[strat_b]
            print(f"  {name_a:15s} vs {name_b:15s}: {wins_a/total*100:.0f}% / {wins_b/total*100:.0f}% (n={total})")


if __name__ == "__main__":
    main()
