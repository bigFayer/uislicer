# UISlicer

基于 **Grounding DINO + SAM 2** 的游戏UI元素自动切片工具。用自然语言描述要切的元素类型，自动检测、像素级分割、导出透明PNG。

## 特性

- **文本引导检测** — 用 `--prompt "coin . diamond . button"` 描述目标元素
- **像素级掩码** — SAM 2 生成精确分割，圆形/不规则形状都能裁切
- **透明PNG导出** — 掩码外区域自动变透明，距离变换羽化抗锯齿
- **NMS去重** — 检测阶段自动去除重叠框，减少冗余推理
- **批处理 + 多掩码择优** — SAM 2 批量推理，3掩码选最优
- **低显存友好** — Grounding DINO 和 SAM 2 顺序加载，2GB显存即可运行
- **后处理脚本** — 自动去误检、去重、重命名为规范文件名

## 环境要求

- NVIDIA GPU（CUDA）
- Python 3.10+
- PyTorch CUDA 版

## 安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. 安装 PyTorch（CUDA 版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 安装依赖
pip install -r requirements.txt
```

> 首次运行会自动下载模型文件（Grounding DINO ~600MB + SAM 2 ~80MB）

## 使用

### 基础用法

```bash
python ui_slicer_pro.py "ui_image.png"
```

### 自定义参数

```bash
python ui_slicer_pro.py "ui_image.png" \
  --prompt "coin . diamond . star . button . icon . chest" \
  --output sliced_output \
  --box_threshold 0.15 \
  --text_threshold 0.15 \
  --padding 4
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--prompt` / `-p` | 检测提示词，用 `.` 分隔元素类型 | 内置游戏UI通用提示 |
| `--output` / `-o` | 输出目录 | `<图片名>_sliced` |
| `--padding` / `-pd` | 裁切外扩边距（px） | 4 |
| `--box_threshold` / `-bt` | 检测框置信度阈值，越低检测越多 | 0.3 |
| `--text_threshold` / `-tt` | 文本匹配阈值 | 0.25 |

### 后处理（去误检 + 去重 + 重命名）

```bash
python post_process.py "ui_image_sliced"
```

自动执行：
1. 删除大面积误检（>10%源图面积 或 宽度>50%源图）
2. IoU去重（重叠>40%保留最佳）
3. 重命名为 `类别_编号.png`（如 `diamond_01.png`, `coin_01.png`）

输出到 `<切片目录>/clean/`。

## 输出示例

```
bg_main_sliced/
├── clean/                  # 后处理后的最终切片
│   ├── diamond_01.png
│   ├── diamond_02.png
│   ├── coin_01.png
│   ├── circle_01.png
│   ├── reward_01.png
│   ├── manifest.json       # 元素清单（坐标、分类、IoU）
│   └── ...
├── manifest.json           # 原始检测清单
└── *.png                   # 原始切片
```

## 项目结构

```
├── ui_slicer_pro.py        # 主工具：检测 + 分割 + 导出
├── post_process.py         # 后处理：去误检/去重/重命名
├── requirements.txt        # Python 依赖
└── docs/
    └── guide.html          # 详细指南
```

## 技术架构

```
输入图片
   │
   ▼
┌──────────────────┐
│  Grounding DINO  │  文本引导检测 → 边界框
│  (Phase 1)       │  + NMS 去重
└──────────────────┘
   │ 释放显存
   ▼
┌──────────────────┐
│     SAM 2        │  批量推理，3掩码择优
│  (Phase 2)       │  → 像素级掩码
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  导出 (Phase 3)  │  距离变换羽化抗锯齿
│                  │  → 透明PNG切片
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  post_process    │  去误检 + 去重 + 重命名
└──────────────────┘
```

## License

MIT
