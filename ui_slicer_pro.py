#!/usr/bin/env python3
"""
UI切图专业工具 v2.0 - Grounding DINO + SAM 2
==========================================
基于语义分割的像素级UI元素自动切片工具

核心优势（vs 旧版AI坐标方案）：
  - 文本引导检测：用自然语言描述要切的元素类型
  - 像素级掩码：圆形金币、不规则装饰都能精确裁切
  - 透明PNG导出：掩码外区域自动变透明
  - 顺序加载：Grounding DINO 和 SAM 2 不会同时驻留显存，2GB即可跑

用法：
  python ui_slicer_pro.py "图片路径" [--prompt "button . icon . coin"] [--output 输出目录]
"""

import os
import re
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image

# 游戏UI常见元素的默认文本提示
DEFAULT_PROMPT = (
    "button . icon . coin . gold . panel . badge . label . bar . "
    "checkbox . close button . arrow . star . gem . chest . reward . "
    "avatar . decoration . frame . card . tab . header"
)


def nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制：去除重叠检测框"""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return np.array(keep)


def check_env():
    """检查运行环境"""
    print("=" * 60)
    print("  UI切图专业工具 v2.0")
    print("  Grounding DINO + SAM 2 语义分割")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\n[错误] 未检测到CUDA GPU!")
        print("  安装CUDA版PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"  GPU:     {gpu}")
    print(f"  显存:    {vram:.1f} GB")
    print(f"  CUDA:    {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    # 检查依赖
    missing = []
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import sam2
    except ImportError:
        missing.append("sam2")
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    if missing:
        print(f"[错误] 缺少依赖: {', '.join(missing)}")
        print(f"  安装: pip install {' '.join(missing)}")
        sys.exit(1)


def phase1_detect(image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
    """
    Phase 1: Grounding DINO 文本引导检测
    输入图片 + 文本提示（如 "button . icon . coin"），返回精确边界框
    """
    print("[Phase 1] Grounding DINO 文本引导检测")
    print("-" * 40)

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "cuda"
    model_id = "IDEA-Research/grounding-dino-tiny"

    print(f"  加载模型: {model_id} (首次运行需下载，约600MB)")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print(f"  模型就绪: {time.time() - t0:.1f}s")

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    print(f"  图片: {w}x{h}")
    print(f"  检测提示: {text_prompt}")

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"  推理完成: {time.time() - t0:.1f}s")

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )

    boxes = results[0]["boxes"].cpu().numpy()  # (N, 4) xyxy
    labels = list(results[0]["labels"])
    scores = results[0]["scores"].cpu().numpy()

    print(f"  检测到 {len(labels)} 个元素:")
    for i, (label, score) in enumerate(zip(labels, scores)):
        x1, y1, x2, y2 = boxes[i]
        print(f"    [{i + 1}] {label}  conf={score:.2f}  bbox=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")

    # NMS去重
    keep = nms(boxes, scores, iou_threshold=0.5)
    boxes = boxes[keep]
    labels = [labels[i] for i in keep]
    scores = scores[keep]
    print(f"  NMS去重后: {len(labels)} 个元素")

    # 释放显存
    del model, processor, inputs, outputs
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
    print(f"  [显存释放] 当前占用: {alloc:.2f} GB")
    print()

    return boxes, labels, scores


def phase2_segment(image_path, boxes):
    """
    Phase 2: SAM 2 像素级分割
    接收边界框，为每个框生成像素级二值掩码
    """
    print("[Phase 2] SAM 2 像素级分割")
    print("-" * 40)

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda"

    print(f"  加载模型: sam2.1-hiera-tiny (首次运行需下载，约80MB)")
    t0 = time.time()
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2.1-hiera-tiny",
        device=device,
    )
    print(f"  模型就绪: {time.time() - t0:.1f}s")

    image = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(image)

    all_masks = []
    all_scores = []

    t0 = time.time()
    # 批量推理 + 多掩码择优
    batch_masks, batch_scores, _ = predictor.predict(
        box=boxes,
        multimask_output=True,
    )
    # batch_masks: (N, 3, H, W), batch_scores: (N, 3)
    # 选每个框IoU最高的掩码
    best_idx = batch_scores.argmax(axis=1)
    all_masks = [batch_masks[i, best_idx[i]] for i in range(len(boxes))]
    all_scores = [float(batch_scores[i, best_idx[i]]) for i in range(len(boxes))]
    elapsed = time.time() - t0

    print(f"  分割完成: {len(all_masks)} 个掩码 ({elapsed:.1f}s)")
    for i, score in enumerate(all_scores):
        print(f"    [{i + 1}] IoU={score:.3f}")

    # 释放显存
    del predictor
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
    print(f"  [显存释放] 当前占用: {alloc:.2f} GB")
    print()

    return np.array(all_masks), np.array(all_scores)


def phase3_export(image_path, masks, labels, boxes, scores, output_dir, padding=4):
    """
    Phase 3: 导出透明PNG切片
    将每个掩码区域裁切为独立PNG文件
    """
    print("[Phase 3] 导出切片")
    print("-" * 40)

    img = np.array(Image.open(image_path).convert("RGBA"))
    os.makedirs(output_dir, exist_ok=True)

    name_counter = {}
    slices = []

    for i, (mask, label, box, score) in enumerate(zip(masks, labels, boxes, scores)):
        # 距离变换羽化：按掩码边缘距离做alpha渐变，比高斯模糊更锐利
        mask_u8 = mask.astype(np.uint8)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        alpha = np.clip(dist / 2.0, 0, 1)
        alpha[mask_u8 == 0] = 0
        element = img.copy()
        element[:, :, 3] = (alpha * 255).astype(np.uint8)

        # 找掩码实际占据的区域（裁掉空白）
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            print(f"  跳过空掩码: {label}_{i}")
            continue

        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        y1, y2 = y_idx[0], y_idx[-1] + 1
        x1, x2 = x_idx[0], x_idx[-1] + 1

        # 加padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)

        cropped = element[y1:y2, x1:x2]

        # 跳过过小区域
        if cropped.shape[0] < 5 or cropped.shape[1] < 5:
            continue

        # 生成唯一文件名
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', label)
        if safe_name in name_counter:
            name_counter[safe_name] += 1
            fname = f"{safe_name}_{name_counter[safe_name]}"
        else:
            name_counter[safe_name] = 0
            fname = safe_name

        out_path = os.path.join(output_dir, f"{fname}.png")
        Image.fromarray(cropped).save(out_path)

        slices.append({
            "file": f"{fname}.png",
            "label": label,
            "bbox": {
                "x": int(x1), "y": int(y1),
                "width": int(x2 - x1), "height": int(y2 - y1),
            },
            "mask_iou": round(float(score), 3),
            "size": f"{cropped.shape[1]}x{cropped.shape[0]}",
        })

        print(f"  [{i + 1}/{len(masks)}] {fname}.png "
              f"({cropped.shape[1]}x{cropped.shape[0]}) IoU={score:.2f}")

    return slices


def generate_preview(image_path, slices, masks, output_dir):
    """生成检测框+掩码叠加预览图"""
    # 用numpy读取以支持中文路径
    img_buf = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)

    # 按label分配颜色
    color_palette = [
        (0, 200, 255), (0, 255, 100), (255, 100, 0), (0, 255, 255),
        (255, 0, 255), (200, 200, 0), (0, 100, 255), (255, 200, 0),
        (128, 0, 255), (0, 128, 255), (255, 128, 0), (128, 255, 0),
    ]
    color_map = {}
    for s in slices:
        if s["label"] not in color_map:
            color_map[s["label"]] = color_palette[len(color_map) % len(color_palette)]

    # 绘制半透明掩码叠加
    overlay = img.copy()
    for idx, s in enumerate(slices):
        mask = masks[idx].astype(bool)
        color = color_map[s["label"]]
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # 绘制边界框和标签
    for s in slices:
        bx, by = s["bbox"]["x"], s["bbox"]["y"]
        bw, bh = s["bbox"]["width"], s["bbox"]["height"]
        color = color_map[s["label"]]
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)
        text = f"{s['label']} {s['size']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (bx, by - th - 6), (bx + tw + 4, by), color, -1)
        cv2.putText(img, text, (bx + 2, by - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    preview_path = os.path.join(output_dir, "_preview.png")
    # 用PIL保存以支持中文路径
    preview_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(preview_rgb).save(preview_path)
    print(f"  预览图: {preview_path}")


def generate_manifest(image_path, slices, output_dir, text_prompt):
    """生成JSON清单文件"""
    w, h = Image.open(image_path).size

    categories = {}
    for s in slices:
        categories[s["label"]] = categories.get(s["label"], 0) + 1

    manifest = {
        "source": os.path.basename(image_path),
        "source_size": {"width": w, "height": h},
        "method": "Grounding DINO + SAM 2",
        "text_prompt": text_prompt,
        "created_at": datetime.now().isoformat(),
        "total_slices": len(slices),
        "categories": categories,
        "slices": slices,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  清单: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="UI切图专业工具 v2.0 - Grounding DINO + SAM 2 语义分割"
    )
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument(
        "--prompt", "-p", default=DEFAULT_PROMPT,
        help="检测提示词，用 . 分隔不同元素类型",
    )
    parser.add_argument("--output", "-o", default=None, help="输出目录（默认=图片名_sliced）")
    parser.add_argument("--padding", "-pd", type=int, default=4, help="外扩边距 px（默认4）")
    parser.add_argument("--box_threshold", "-bt", type=float, default=0.3,
                        help="检测框置信度阈值（默认0.3，降低可检测更多元素）")
    parser.add_argument("--text_threshold", "-tt", type=float, default=0.25,
                        help="文本匹配阈值（默认0.25）")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"错误: 图片不存在 - {args.image}")
        sys.exit(1)

    # 自动输出目录
    if args.output is None:
        base = Path(args.image).stem
        parent = Path(args.image).parent
        args.output = str(parent / f"{base}_sliced")

    check_env()

    print(f"  输入: {args.image}")
    print(f"  输出: {args.output}")
    print()

    t_total = time.time()

    # Phase 1: 文本引导检测
    boxes, labels, scores = phase1_detect(
        args.image, args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if len(boxes) == 0:
        print("未检测到任何元素，建议：")
        print("  1. 降低 --box_threshold（如 0.15）")
        print("  2. 更换 --prompt（更具体的描述，如 'gold coin . blue button'）")
        sys.exit(0)

    # Phase 2: 像素级分割
    masks, mask_scores = phase2_segment(args.image, boxes)

    # Phase 3: 导出
    slices = phase3_export(
        args.image, masks, labels, boxes, mask_scores,
        args.output, args.padding,
    )

    # 生成预览和清单
    print("[输出]")
    generate_manifest(args.image, slices, args.output, args.prompt)
    try:
        generate_preview(args.image, slices, masks, args.output)
    except Exception as e:
        print(f"  [跳过预览图: {e}]")

    # 汇总
    elapsed = time.time() - t_total
    cats = {}
    for s in slices:
        cats[s["label"]] = cats.get(s["label"], 0) + 1

    print()
    print("=" * 60)
    print(f"  切图完成!")
    print(f"  耗时:   {elapsed:.1f}s")
    print(f"  输出:   {args.output}")
    print(f"  切片:   {len(slices)} 个")
    print(f"  分类:   {cats}")
    print("=" * 60)


if __name__ == "__main__":
    main()