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

| 项目 | 最低要求 | 推荐 |
|------|----------|------|
| GPU | NVIDIA GPU，2GB+ 显存 | RTX 3060 及以上 |
| CUDA | 11.8+ | 12.1+ |
| Python | 3.10 | 3.11 / 3.12 |
| 磁盘 | ~3GB（模型 + 依赖） | 建议预留 5GB |

> 不支持 AMD / Intel 集显，必须 NVIDIA GPU + CUDA。

## 安装（详细版）

### 第一步：确认 GPU 和 CUDA

```bash
# Windows: 打开 cmd 或 PowerShell
nvidia-smi
```

在输出右上角找到 `CUDA Version: XX.X`，记住这个版本号。

如果没有输出或报错，说明没有 NVIDIA 驱动，先去 [NVIDIA 驱动下载](https://www.nvidia.com/Download/index.aspx) 安装。

### 第二步：安装 Python

如果还没有 Python 3.10+，去 [python.org](https://www.python.org/downloads/) 下载安装。

**Windows 安装时务必勾选 `Add Python to PATH`**。

验证：
```bash
python --version
# 应显示 Python 3.10.x 或更高
```

### 第三步：创建虚拟环境

```bash
# 进入项目目录
cd uislicer

# 创建虚拟环境
python -m venv venv

# 激活
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

> **C盘空间不足？** 把虚拟环境建到其他盘：
> ```bash
> python -m venv D:\slicer_env
> D:\slicer_env\Scripts\activate
> ```

### 第四步：安装 PyTorch（CUDA 版）

根据第一步查到的 CUDA 版本选择对应的安装命令：

**CUDA 12.8（最新）：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**CUDA 12.4：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 11.8（旧卡）：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

不确定版本？查看[PyTorch 官网](https://pytorch.org/get-started/locally/)选对应的命令。

验证安装：
```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

应输出：
```
CUDA可用: True, GPU: NVIDIA GeForce RTX XXXX
```

如果 `CUDA可用: False`，说明 PyTorch 的 CUDA 版本和驱动不匹配，重新按正确版本安装。

### 第五步：安装其余依赖

```bash
pip install -r requirements.txt
```

> 首次运行工具时会自动下载模型：
> - Grounding DINO (`grounding-dino-tiny`) ~600MB
> - SAM 2 (`sam2.1-hiera-tiny`) ~80MB
>
> 模型缓存在 `~/.cache/huggingface/`，下载一次后不需要重复下载。

## 使用

### 基础用法

```bash
python ui_slicer_pro.py "ui_image.png"
```

自动输出到 `ui_image_sliced/` 目录。

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

### 提示词技巧

`--prompt` 是影响检测效果的关键参数：

```bash
# 精简提示（只要特定类型）:
--prompt "coin . diamond . star"

# 泛化提示（尽量多检测）:
--prompt "button . icon . coin . gold . panel . badge . label . bar . star . gem . chest . reward . avatar . card . tab"

# 针对性描述（颜色+形状更精准）:
--prompt "purple diamond . gold coin . blue button . red star"

# 英文句点 . 是分隔符，每个词独立匹配
```

**检测太少？** 降低 `--box_threshold` 到 0.15 或 0.10。
**检测太多/误检？** 提高 `--box_threshold` 到 0.35 或 0.40。

### 后处理（去误检 + 去重 + 重命名）

```bash
python post_process.py "ui_image_sliced"
```

自动执行：
1. 删除大面积误检（>10%源图面积 或 宽度>50%源图）
2. IoU去重（重叠>40%保留最佳）
3. 重命名为 `类别_编号.png`（如 `diamond_01.png`, `coin_01.png`）

输出到 `<切片目录>/clean/`。

可以在 `post_process.py` 中修改 `LABEL_MAP` 字典来自定义分类名称映射。

## 完整工作流示例

```bash
# 1. 激活环境
venv\Scripts\activate

# 2. 切图（低阈值检测更多元素）
python ui_slicer_pro.py "bg_main.png" \
  -p "coin . gold coin . diamond . star . reward . gift . card . icon . badge . label" \
  -bt 0.15 -tt 0.15

# 3. 后处理（自动清理 + 重命名）
python post_process.py "bg_main_sliced"

# 4. 最终切片在 clean/ 目录
ls bg_main_sliced/clean/
# diamond_01.png  diamond_02.png  coin_01.png  ...
```

## 输出说明

```
bg_main_sliced/
├── clean/                  # ← 最终使用的切片（后处理后）
│   ├── diamond_01.png
│   ├── coin_01.png
│   ├── star_01.png
│   ├── reward_01.png
│   ├── manifest.json       # 元素清单（坐标、分类、IoU）
│   └── ...
├── manifest.json           # 原始检测清单
└── *.png                   # 原始切片（未清理）
```

`manifest.json` 内容：
```json
{
  "source": "bg_main.png",
  "source_size": {"width": 855, "height": 1338},
  "total_slices": 31,
  "categories": {"diamond": 6, "coin": 5, "circle": 6, ...},
  "slices": [
    {
      "file": "diamond_01.png",
      "category": "diamond",
      "bbox": {"x": 273, "y": 355, "width": 84, "height": 83},
      "mask_iou": 0.982,
      "size": "84x83"
    }
  ]
}
```

## 项目结构

```
├── ui_slicer_pro.py        # 主工具：检测 + 分割 + 导出
├── post_process.py         # 后处理：去误检/去重/重命名
├── requirements.txt        # Python 依赖
├── bg_main.png             # 示例图片
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

## 常见问题

### `CUDA不可用` / `torch.cuda.is_available() = False`

PyTorch 安装的是 CPU 版本而非 CUDA 版本。重新安装：

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

确认 CUDA 版本号和 `nvidia-smi` 显示的一致。

### `pip install` 下载慢

使用国内镜像：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### C盘空间不足

pip 默认缓存到 C 盘。解决方案：

```bash
# 把虚拟环境建到其他盘
python -m venv E:\slicer_env

# pip 缓存也指向其他盘
pip install --cache-dir E:\pip_cache torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### `OSError` / 中文路径报错

确保文件路径不含特殊字符，或使用绝对路径：

```bash
# 用绝对路径
python ui_slicer_pro.py "D:\images\ui.png" -o "D:\output\sliced"
```

### 检测到的元素太少

1. 降低阈值：`--box_threshold 0.10 --text_threshold 0.10`
2. 换更具体的提示词：`--prompt "purple diamond . gold coin . blue button"`
3. 检查图片尺寸是否过大（>2000px 建议先缩小）

### 检测到太多误检

1. 提高阈值：`--box_threshold 0.35`
2. 使用更精确的提示词
3. 运行后处理脚本自动清理：`python post_process.py "输出目录"`

### 显存不够 (OOM)

本工具已做显存优化（Grounding DINO 和 SAM 2 不会同时驻留显存）。如果仍 OOM：

- 关闭其他占用 GPU 的程序（浏览器、游戏、其他 AI 工具）
- 2GB 显存即可运行（已实测）

## License

MIT
