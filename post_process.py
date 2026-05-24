#!/usr/bin/env python3
"""
切图后处理 - 去误检、去重、重命名
用法: python post_process.py [切图输出目录]
"""

import sys
import json
import shutil
from pathlib import Path

# 标签 → 简洁分类名
LABEL_MAP = {
    "purple diamond": "diamond",
    "diamond icon": "diamond",
    "coin gold coin": "coin",
    "coin": "coin",
    "gold coin": "coin",
    "gold": "gold",
    "circle yellow circle": "circle",
    "circle yellow": "circle",
    "circle": "circle",
    "gold yellow circle": "circle",
    "gold coin circle yellow circle": "circle",
    "reward icon icon": "reward",
    "star": "star",
    "star diamond icon": "star",
    "gift": "gift",
    "box gift": "gift_box",
    "card": "card",
    "label": "label",
    "chest": "chest",
    "chest box": "chest",
    "box": "box",
}


def compute_iou(a, b):
    """计算两个bbox的IoU"""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = a["width"] * a["height"]
    b_area = b["width"] * b["height"]
    return inter / (a_area + b_area - inter)


def main():
    if len(sys.argv) > 1:
        sliced_dir = Path(sys.argv[1])
    else:
        sliced_dir = Path(__file__).parent / "bg_main_sliced"

    manifest_path = sliced_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"错误: 找不到 {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    src_w = manifest["source_size"]["width"]
    src_h = manifest["source_size"]["height"]
    src_area = src_w * src_h

    slices = manifest["slices"]
    print(f"原始切片: {len(slices)} 个")
    print()

    # --- Step 1: 过滤大面积误检 ---
    print("[Step 1] 过滤大面积误检")
    kept = []
    for s in slices:
        bbox = s["bbox"]
        area_ratio = bbox["width"] * bbox["height"] / src_area
        width_ratio = bbox["width"] / src_w

        if area_ratio > 0.10 or width_ratio > 0.50:
            print(f"  删除 {s['file']} ({s['size']}, 面积{area_ratio:.0%}, 宽{width_ratio:.0%})")
        else:
            kept.append(s)

    print(f"  保留 {len(kept)} 个 (删除 {len(slices) - len(kept)})")
    print()

    # --- Step 2: IoU去重 ---
    print("[Step 2] 坐标去重")
    deduped = []
    for s in kept:
        is_dup = False
        for i, k in enumerate(deduped):
            iou = compute_iou(s["bbox"], k["bbox"])
            if iou > 0.4:
                # 保留mask_iou更高的
                if s["mask_iou"] > k["mask_iou"]:
                    print(f"  替换: {k['file']} <- {s['file']} (IoU={iou:.2f})")
                    deduped[i] = s
                else:
                    print(f"  去重: {s['file']} = {k['file']} (IoU={iou:.2f})")
                is_dup = True
                break
        if not is_dup:
            deduped.append(s)

    print(f"  保留 {len(deduped)} 个 (去重 {len(kept) - len(deduped)})")
    print()

    # --- Step 3: 重命名导出到 clean/ ---
    print("[Step 3] 重命名导出")
    clean_dir = sliced_dir / "clean"
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir()

    counter = {}
    for s in deduped:
        cat = LABEL_MAP.get(s["label"], s["label"].replace(" ", "_"))
        counter.setdefault(cat, 0)
        counter[cat] += 1
        new_name = f"{cat}_{counter[cat]:02d}.png"

        src_file = sliced_dir / s["file"]
        if src_file.exists():
            shutil.copy2(src_file, clean_dir / new_name)

        old_name = s["file"]
        s["file"] = new_name
        s["category"] = cat
        print(f"  {old_name} -> {new_name}")

    # 写入新清单
    new_manifest = {
        "source": manifest["source"],
        "source_size": manifest["source_size"],
        "method": manifest["method"] + " + 后处理",
        "total_slices": len(deduped),
        "categories": counter,
        "slices": deduped,
    }

    with open(clean_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 50)
    print(f"  后处理完成!")
    print(f"  {len(slices)} -> {len(deduped)} 个切片")
    print(f"  输出: {clean_dir}")
    print(f"  分类: {counter}")
    print("=" * 50)


if __name__ == "__main__":
    main()
