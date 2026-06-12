#!/usr/bin/env python3
"""测试 genai 环境：GPU、模型加载、推理"""

import torch
import time
import sys

print("=" * 60)
print("1️⃣  PyTorch + GPU 基础测试")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"SM 支持: {torch.cuda.get_arch_list()}")

# GPU 矩阵乘法性能测试
if torch.cuda.is_available():
    print("\n📊 GPU 矩阵乘法性能测试 (4096x4096)...")
    a = torch.randn(4096, 4096).cuda()
    b = torch.randn(4096, 4096).cuda()
    
    # 预热
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    
    # 正式测试
    t0 = time.time()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    t = time.time() - t0
    print(f"  10 次矩阵乘法耗时: {t:.3f}s ({t/10*1000:.1f}ms/次)")

print("\n" + "=" * 60)
print("2️⃣  Transformers 加载小模型测试")
print("=" * 60)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import accelerate
    print(f"transformers 版本: {__import__('transformers').__version__}")
    print(f"accelerate 版本: {accelerate.__version__}")
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n加载模型: {model_name}")
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    t = time.time() - t0
    print(f"  加载耗时: {t:.1f}s")
    print(f"  模型参数量: {model.num_parameters() / 1e6:.1f}M")
    print(f"  模型设备: {model.device}")
    
    # 推理测试
    print("\n📝 推理测试:")
    messages = [{"role": "user", "content": "请用一句话介绍深度学习"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
    )
    t = time.time() - t0
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  生成耗时: {t:.2f}s ({outputs.shape[1] - inputs.input_ids.shape[1]} tokens)")
    print(f"  速度: {(outputs.shape[1] - inputs.input_ids.shape[1]) / t:.1f} tokens/s")
    print(f"  输出: {response}")

except Exception as e:
    print(f"  ❌ 测试失败: {e}")

print("\n" + "=" * 60)
print("3️⃣  Diffusers 测试（文生图）")
print("=" * 60)

try:
    from diffusers import StableDiffusionPipeline
    import diffusers
    print(f"diffusers 版本: {diffusers.__version__}")
    
    # 用个很小的模型测试
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    print(f"\n加载模型: {model_id}")
    
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    t = time.time() - t0
    print(f"  加载耗时: {t:.1f}s")
    
    print("\n🖼️  生成图片测试:")
    t0 = time.time()
    image = pipe(
        "a cute cat",
        num_inference_steps=20,
        width=256,
        height=256,
    ).images[0]
    t = time.time() - t0
    print(f"  生成耗时: {t:.2f}s")
    print(f"  图片尺寸: {image.size}")

except Exception as e:
    print(f"  ❌ 测试失败: {e}")

print("\n" + "=" * 60)
print("4️⃣  PEFT/QLoRA 测试")
print("=" * 60)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    import peft
    print(f"peft 版本: {peft.__version__}")
    
    from transformers import AutoModelForCausalLM
    
    print("\n加载小模型进行 LoRA 测试...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("  ✅ LoRA 配置成功")

except Exception as e:
    print(f"  ❌ 测试失败: {e}")

print("\n" + "=" * 60)
print("5️⃣  bitsandbytes 量化测试")
print("=" * 60)

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes 版本: {bnb.__version__}")
    
    # 测试 4bit 量化
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    print("\n加载 4bit 量化模型...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )
    t = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  加载耗时: {t:.1f}s")
    print(f"  峰值显存: {mem:.2f} GB")
    print(f"  模型类型: {model.dtype}")
    print("  ✅ 4bit 量化成功")

except Exception as e:
    print(f"  ❌ 测试失败: {e}")

print("\n" + "=" * 60)
print("✅ 所有测试完成！")
print("=" * 60)
