"""
单张图片对推理：输入两张卫星图，输出变化描述
使用 InternVL3-1B 本地模型（仅需 1.8GB 显存）

用法：
    conda activate genai
    python inference/predict_one.py --pre /path/to/pre.png --post /path/to/post.png
"""

import argparse, os, torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from torchvision.transforms.functional import InterpolationMode

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def build_transform(input_size=448):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    return Compose([
        Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = orig_w * orig_h
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized = image.resize((target_w, target_h))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def load_image(image_file, input_size=448, max_num=6):
    if isinstance(image_file, Image.Image):
        image = image_file.convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


def main():
    parser = argparse.ArgumentParser(description="单张卫星图像变化描述推理（InternVL3-1B）")
    parser.add_argument("--pre", required=True, help="灾前图像路径")
    parser.add_argument("--post", required=True, help="灾后图像路径")
    parser.add_argument("--prompt", default="textual",
                        choices=["zero-shot", "textual", "visual"],
                        help="prompt 策略")
    parser.add_argument("--max_tokens", type=int, default=256, help="最大生成 token 数")
    args = parser.parse_args()

    for p in [args.pre, args.post]:
        if not os.path.exists(p):
            print(f"错误: 找不到图像 {p}")
            return

    # 加载模型
    print("加载 InternVL3-1B...")
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL3-1B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL3-1B",
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # 加载图片
    print("加载图片...")
    pixel_values1 = load_image(args.pre).to("cuda:0", torch.bfloat16)
    pixel_values2 = load_image(args.post).to("cuda:0", torch.bfloat16)
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    # Prompt
    text_prompt = {
        "zero-shot": "Give change description between two satellite images. Output answer in a news style with a few sentences using precise phrases separated by commas.",
        "textual": "These two satellite images show a natural disaster. Here is the disaster level descriptions: - Disaster Level 0 (No Damage): Undisturbed. - Disaster Level 1 (Minor Damage): Building partially burnt, roof elements missing. - Disaster Level 2 (Major Damage): Partial wall or roof collapse. - Disaster Level 3 (Destroyed): Scorched, completely collapsed. Now, describe the changes that occurred between the pre-event and post-event images with the given disaster level descriptions. Output answer in a news style with a few sentences using precise phrases separated by commas.",
        "visual": "These two satellite images show a natural disaster. Here is the disaster level descriptions: - Disaster Level 0 (No Damage): Undisturbed. - Disaster Level 1 (Minor Damage): Building partially burnt, roof elements missing. - Disaster Level 2 (Major Damage): Partial wall or roof collapse. - Disaster Level 3 (Destroyed): Scorched, completely collapsed. Now, describe the changes that occurred between the pre-event and post-event images with the given disaster level descriptions. Output answer in a news style with a few sentences using precise phrases separated by commas.",
    }[args.prompt]

    question = f"<image>\n<image>\n{text_prompt}"

    # 推理
    print("生成描述中...")
    generation_config = dict(max_new_tokens=args.max_tokens, do_sample=False)
    with torch.no_grad():
        response = model.chat(
            tokenizer, pixel_values, question,
            generation_config, num_patches_list=num_patches_list
        )

    print("\n" + "=" * 50)
    print(f"模型: InternVL3-1B (本地)")
    print(f"显存占用: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    print(f"Prompt 策略: {args.prompt}")
    print(f"灾前: {args.pre}")
    print(f"灾后: {args.post}")
    print("=" * 50)
    print(f"变化描述:\n{response}")
    print("=" * 50)


if __name__ == "__main__":
    main()
