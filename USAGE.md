# 使用指南

## 环境要求

- Python 3.10（conda 环境 `genai`）
- CUDA 12.x
- 显存建议：≥ 24GB（3B 模型约需 7GB + 图片处理开销）

## 1. 环境配置

```bash
# 使用现有 conda 环境
conda activate genai

# 环境文件位于
cat environment.yml

# 关键依赖版本
pip show transformers   # 4.51.3
pip show torch          # 2.6.0+cu124
```

## 2. 数据准备

xBD 数据集子集存放在：
```
RSCC-Data/xbd_subset/
├── {event_name}/
│   ├── images/           # 切分后的卫星图像（512×512 切片）
│   └── labels/           # 标注 JSON（建筑坐标 + 损伤等级）
```

图像命名规则：
```
{event_name}_{id}_pre_disaster_part{1-4}.png   # 灾前图像
{event_name}_{id}_post_disaster_part{1-4}.png  # 灾后图像
```

每张原图被切分为 4 个 part（左上、右上、左下、右下）。

## 3. 模型推理

### 3.1 单模型推理（推荐用于实验）

**支持的模型**（已测试）：
- `Qwen/Qwen2.5-VL-3B-Instruct` ✅
- `OpenGVLab/InternVL3-1B` ✅
- `OpenGVLab/InternVL3-2B` ✅
- `OpenGVLab/InternVL3-8B` ✅

**使用方法**：

```bash
# 修改 inference/xbd_subset_baseline.py 顶部 INFERENCE_MODEL_LIST
# 改为需要的模型名称

python inference/xbd_subset_baseline.py \
  --output_file ./output/test.jsonl \
  --log_file ./logs/test.log \
  --device cuda:0
```

推理脚本会自动：
1. 扫描所有图片对
2. 解析标签文件获取灾害类型和建筑损伤统计
3. 使用三种 prompt 策略分别推理
4. 自动断点续跑（跳过已处理的条目）
5. 结果实时写入 JSONL 文件

### 3.2 三种 Prompt 策略

| 策略 | 文件中的键名 | 说明 |
|:---|:---|:---|
| Zero-shot | `zero-shot` | 直接描述两张图片的变化 |
| Textual Prompt | `textual_prompt` | 提供灾害类型 + 建筑数量 + 损伤等级文字 |
| Visual Prompt | `visual_prompt` | 在上述基础上，告知建筑颜色编码 |

### 3.3 纠错解码（需额外配置）

纠错解码代码在 `inference_with_cd/` 目录，支持三种方法：

- **DoLa**：Transformers 原生支持，需 `dola_layers='high'`
- **DeCo**：需要额外显存（3B 模型需 > 8GB）
- **VCD**：需要双路 forward（3B 模型需 > 10GB）

由于当前硬件（RTX 5060, 7.5GB VRAM）限制，纠错解码运行失败。

## 4. 评估

### 4.1 自动评估

```bash
# 运行完整评估（BLEU、ROUGE、METEOR、语义相似度）
python evaluation/evaluate_all.py \
  --ground_truth ./output/full_qvq.jsonl \
  --predictions ./output/test.jsonl \
  --output_dir ./output/evaluation_results
```

评估指标说明：
- **语义相似度**：使用 sentence-t5-xxl 计算语义嵌入的余弦相似度
- **BLEU-1**：unigram 精确匹配
- **ROUGE-L**：最长公共子序列
- **METEOR**：基于 WordNet 同义词匹配

### 4.2 AutoEval（专家裁判）

使用 Qwen3-VL-Plus 作为裁判，比较两种策略的输出优劣：

```bash
python evaluation/autoeval_3way.py \
  --input ./output/test.jsonl \
  --output_dir ./output/evaluation_results/autoeval \
  --max_samples 100 \
  --workers 3
```

需要配置 `.env` 文件：
```
DASHSCOPE_API_KEY="sk-xxxx"
QVQ_MODEL_NAME="qwen3-vl-plus"
API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## 5. 数据集

### 5.1 xBD 子集

训练（`split=train`）和测试（`split=test`）划分由标签文件中的 `split` 字段确定。

数据集包含 12 类灾害事件：

| 灾害类型 | 事件 |
|:---|:---|
| 火山爆发 | guatemala-volcano, lower-puna-volcano |
| 野火 | pinery-bushfire, portugal-wildfire, santa-rosa-wildfire, socal-fire, woolsey-fire |
| 风暴 | hurricane-florence, joplin-tornado, moore-tornado, hurricane-matthew, tuscaloosa-tornado |
| 地震 | mexico-earthquake |
| 洪水 | hurricane-michael, hurricane-harvey, midwest-flooding, nepal-flooding |
| 海啸 | palu-tsunami, sunda-tsunami |

### 5.2 损伤等级

| 等级 | 标签 | 描述 |
|:---|:---|:---|
| 0 | `no-damage` | 无损伤 |
| 1 | `minor-damage` | 轻微损伤（屋顶缺失、裂缝） |
| 2 | `major-damage` | 严重损伤（部分墙体/屋顶倒塌） |
| 3 | `destroyed` | 完全摧毁 |

## 6. 输出格式

推理结果 JSONL，每行包含：

```json
{
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "inference_type": "textual_prompt",
  "pre_image": "/path/to/pre_disaster_part1.png",
  "post_image": "/path/to/post_disaster_part1.png",
  "change_caption": "The area experienced severe damage...",
  "target_caption": "Residential buildings were destroyed..."  // 仅在 eval JSONL 中
}
```

## 7. 结果目录结构

```
output/
├── test.jsonl                           # 推理结果（3 种策略 × 988 条 = 2964 条）
├── full_qvq.jsonl                       # Ground Truth（QvQ-Max 生成）
├── inference_cd.jsonl                   # 纠错解码结果
├── evaluation_results/
│   ├── full_metrics_final.json          # 自动评估指标汇总
│   ├── metrics_results.json             # 详细指标
│   ├── self_eval_report.json            # 关键词/多样性分析
│   ├── win_rates.json                   # AutoEval 胜负率
│   ├── 实验报告.md                       # 完整实验报告
│   ├── pred_zero-shot.jsonl             # Zero-shot 预测
│   ├── pred_textual_prompt.jsonl        # Textual Prompt 预测
│   ├── pred_visual_prompt.jsonl         # Visual Prompt 预测
│   ├── best_model_results_qvq.jsonl     # GT vs Visual 对比
│   └── autoeval/                        # AutoEval 详细结果
│       ├── results_zero-shot_vs_textual_prompt.jsonl
│       ├── results_zero-shot_vs_visual_prompt.jsonl
│       └── results_textual_prompt_vs_visual_prompt.jsonl
└── evaluation_results_qvq/              # 旧版评估结果
```

## 8. 常见问题

**Q: CUDA OOM (Out of Memory)**
A: 7.5GB VRAM 仅能运行 3B 模型（bfloat16 需约 6GB）。
- 尝试 `attn_implementation="eager"` 而非 `flash_attention_2`
- 使用 batch_size=1
- 关闭纠错解码（DoLa/DeCo/VCD 需要额外显存）

**Q: 模型加载失败**
A: 确保 `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` 并使用 `local_files_only=True`。
模型缓存位于 `~/.cache/huggingface/hub/`。

**Q: 如何添加新模型支持？**
A: 在 `utils/model_hub.py` 中添加新的 `elif` 分支，实现模型加载逻辑，并在 `utils/constants.py` 的 `MODEL_LIST` 中添加模型名。

**Q: JSONL 文件格式**
A: 每行一个 JSON 对象。支持断点续跑（自动跳过已处理的 pre_image + post_image + model_id 组合）。
