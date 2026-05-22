#!/usr/bin/env python3
"""
LFW 闭集 1:N 人脸识别业务测试报告。

测试口径：
- 从 LFW funneled 目录中筛选每人图片数 >= min-images-per-person 的身份。
- 每个身份选 1 张成功提取向量的图片作为底库注册照。
- 其余图片全部作为查询图，逐张调用远端人脸服务提取向量。
- 每张查询图与全部底库向量计算余弦相似度，统计 Rank-1/5/10 命中率。
- 生成中文 HTML 报告，展示总体指标、CMC、按人统计、Top10 可视化案例和向量缩略图。

脚本只调用远端 HTTP 接口，不加载本地 face_model.py。
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import math
import mimetypes
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="生成 LFW 闭集 1:N 中文业务测试报告")
    parser.add_argument("--api-url", default="http://10.11.204.167:8188/", help="人脸服务 POST 地址")
    parser.add_argument("--lfw-dir", default="reports/_lfw_cache/lfw_home/lfw_funneled", help="LFW funneled 图片目录")
    parser.add_argument("--out-html", default="reports/lfw_closed_set_business_report.html", help="输出 HTML 路径")
    parser.add_argument("--out-json", default=None, help="输出 JSON 路径，默认与 HTML 同名")
    parser.add_argument("--cache-json", default="reports/lfw_closed_set_feature_cache.json", help="向量缓存 JSON")
    parser.add_argument("--min-images-per-person", type=int, default=20, help="每人至少多少张图，默认 20")
    parser.add_argument("--top-k", type=int, default=10, help="展示和统计的 TopK，默认 10")
    parser.add_argument("--same-threshold", type=float, default=0.55, help="业务建议同人阈值，仅用于报告标注")
    parser.add_argument("--review-threshold", type=float, default=0.45, help="人工复核阈值，仅用于报告标注")
    parser.add_argument("--timeout", type=float, default=30.0, help="单张图片请求超时秒数")
    parser.add_argument("--face-select", choices=("score", "area", "first"), default="score", help="多脸图片选择规则")
    parser.add_argument("--no-cache", action="store_true", help="不读取已有缓存，但仍会写出新缓存")
    parser.add_argument("--refresh-cache", action="store_true", help="忽略缓存，重新请求全部图片")
    parser.add_argument("--save-every", type=int, default=20, help="每处理多少张图片保存一次缓存")
    parser.add_argument("--enroll-index", type=int, default=0, help="优先选每个人排序后的第几张作为底库，默认 0")
    parser.add_argument("--no-enroll-fallback", action="store_true", help="底库候选图失败时不尝试下一张")
    parser.add_argument("--detail-mode", choices=("errors", "all", "none"), default="errors", help="HTML 可视化案例范围")
    parser.add_argument("--max-case-cards", type=int, default=80, help="每类最多展示多少个案例，0 表示不限制")
    parser.add_argument("--success-samples", type=int, default=20, help="Top1 正确样本抽样展示数量")
    parser.add_argument("--image-mode", choices=("file", "embed"), default="file", help="图片引用方式：file 更小，embed 可独立分享但很大")
    parser.add_argument("--seed", type=int, default=20260522, help="成功样本抽样随机种子")
    parser.add_argument("--title", default="LFW 闭集 1:N 人脸识别业务测试报告", help="报告标题")
    parser.add_argument("--header", action="append", default=[], help="额外 HTTP 头，格式 Key:Value，可重复传")
    return parser.parse_args()


def normalize_api_url(url):
    if "://" not in url:
        url = "http://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("--api-url 必须是完整地址，例如 http://10.11.204.167:8188/")
    if parsed.path == "":
        return url.rstrip("/") + "/"
    return url


def parse_headers(header_args):
    headers = {}
    for item in header_args:
        if ":" not in item:
            raise ValueError(f"--header 格式错误: {item}")
        key, value = item.split(":", 1)
        headers[key.strip()] = value.strip()
    return headers


def esc(value):
    return html.escape(str(value), quote=True)


def pct(numerator, denominator):
    if denominator <= 0:
        return "-"
    return f"{numerator / denominator * 100:.2f}%"


def fmt(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def display_name(identity):
    return identity.replace("_", " ")


def scan_lfw(lfw_dir, min_images):
    root = Path(lfw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"LFW 目录不存在: {root}")

    all_people = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if images:
            all_people[person_dir.name] = images

    subset = {name: images for name, images in all_people.items() if len(images) >= min_images}
    subset = dict(sorted(subset.items(), key=lambda item: (-len(item[1]), item[0])))
    return all_people, subset


def image_size(path):
    data = Path(path).read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if data.startswith(b"\xff\xd8"):
        idx = 2
        while idx + 9 < len(data):
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            idx += 2
            if marker in (0xD8, 0xD9):
                continue
            if idx + 2 > len(data):
                break
            length = int.from_bytes(data[idx : idx + 2], "big")
            if length < 2 or idx + length > len(data):
                break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                height = int.from_bytes(data[idx + 3 : idx + 5], "big")
                width = int.from_bytes(data[idx + 5 : idx + 7], "big")
                return width, height
            idx += length
    return None, None


def image_src(path, image_mode):
    path = Path(path)
    if image_mode == "embed":
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    return path.resolve().as_uri()


def load_cache(path, no_cache):
    if no_cache or not Path(path).is_file():
        return {"items": {}}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {"items": {}}
    if not isinstance(data, dict) or not isinstance(data.get("items"), dict):
        return {"items": {}}
    return data


def save_cache(path, cache, api_url):
    cache["api_url"] = api_url
    cache["updated_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def cache_key(path):
    return str(Path(path).resolve())


def cache_valid(entry, path):
    if not entry:
        return False
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        return False
    return entry.get("size") == stat.st_size and int(entry.get("mtime", 0)) == int(stat.st_mtime)


def post_image(api_url, image_path, timeout, extra_headers):
    boundary = "----lfw-closed-set-" + uuid.uuid4().hex
    filename = Path(image_path).name.replace('"', "_")
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_bytes = Path(image_path).read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Accept": "application/json",
    }
    headers.update(extra_headers)
    req = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"请求失败: {exc}") from exc


def bbox_area(bbox):
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def select_face(faces, mode):
    valid = []
    for idx, face in enumerate(faces):
        feature = face.get("feature")
        bbox = face.get("bboxes") or face.get("bbox") or []
        if isinstance(feature, list) and feature:
            valid.append({"index": idx, "face": face, "bbox": bbox, "feature": feature})
    if not valid:
        raise RuntimeError("未检测到可用人脸向量")
    if mode == "first":
        return valid[0]
    if mode == "area":
        return max(valid, key=lambda item: bbox_area(item["bbox"]))
    return max(valid, key=lambda item: float(item["bbox"][4]) if len(item["bbox"]) >= 5 else bbox_area(item["bbox"]))


def normalize_vector(raw):
    vector = [float(x) for x in raw]
    norm = math.sqrt(sum(x * x for x in vector))
    if norm <= 0:
        raise RuntimeError("向量范数为 0")
    return [x / norm for x in vector], norm


def extract_feature(api_url, image_path, args, extra_headers, cache):
    key = cache_key(image_path)
    if not args.refresh_cache:
        cached = cache.get("items", {}).get(key)
        if cache_valid(cached, image_path):
            return cached

    start = time.perf_counter()
    entry = {
        "path": str(Path(image_path)),
        "name": Path(image_path).name,
        "ok": False,
        "error": "",
        "cost_ms": None,
    }
    try:
        data = post_image(api_url, image_path, args.timeout, extra_headers)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data.get("error")))
        if not isinstance(data, list):
            raise RuntimeError(f"接口返回格式不是 list: {type(data).__name__}")
        selected = select_face(data, args.face_select)
        feature, raw_norm = normalize_vector(selected["feature"])
        width, height = image_size(image_path)
        entry.update(
            {
                "ok": True,
                "face_count": len(data),
                "selected_face_index": selected["index"],
                "bbox": selected["bbox"],
                "feature": feature,
                "feature_dim": len(feature),
                "raw_feature_norm": raw_norm,
                "width": width,
                "height": height,
            }
        )
    except Exception as exc:
        entry["error"] = str(exc)
    finally:
        stat = Path(image_path).stat()
        entry["size"] = stat.st_size
        entry["mtime"] = int(stat.st_mtime)
        entry["cost_ms"] = (time.perf_counter() - start) * 1000.0
        cache.setdefault("items", {})[key] = entry
    return entry


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def build_enrollment(subset, api_url, args, extra_headers, cache):
    enrollments = {}
    enrollment_failures = []
    ordered_people = sorted(subset.keys())

    for idx, identity in enumerate(ordered_people, 1):
        images = subset[identity]
        first_idx = min(max(args.enroll_index, 0), len(images) - 1)
        candidate_indexes = [first_idx]
        if not args.no_enroll_fallback:
            candidate_indexes.extend(i for i in range(len(images)) if i != first_idx)

        selected = None
        attempts = []
        print(f"[底库 {idx}/{len(ordered_people)}] {identity}")
        for image_idx in candidate_indexes:
            path = images[image_idx]
            item = extract_feature(api_url, path, args, extra_headers, cache)
            attempts.append({"path": str(path), "ok": item.get("ok"), "error": item.get("error", "")})
            if item.get("ok"):
                selected = dict(item)
                selected["identity"] = identity
                selected["identity_display"] = display_name(identity)
                selected["enroll_image_index"] = image_idx
                break

        if selected:
            enrollments[identity] = selected
            if len(attempts) > 1:
                enrollment_failures.extend(
                    {"identity": identity, "path": a["path"], "error": a["error"]} for a in attempts[:-1] if not a["ok"]
                )
            print(f"  OK {selected['name']}")
        else:
            enrollment_failures.extend(
                {"identity": identity, "path": a["path"], "error": a["error"]} for a in attempts if not a["ok"]
            )
            print("  FAIL 无法注册底库")

    return enrollments, enrollment_failures


def evaluate_queries(subset, enrollments, api_url, args, extra_headers, cache):
    rows = []
    query_failures = []
    all_query_paths = []
    for identity, images in subset.items():
        enroll_path = Path(enrollments[identity]["path"]).resolve() if identity in enrollments else None
        for path in images:
            if enroll_path is not None and path.resolve() == enroll_path:
                continue
            all_query_paths.append((identity, path))

    print(f"[查询] 共 {len(all_query_paths)} 张")
    gallery = list(enrollments.values())
    for idx, (identity, path) in enumerate(all_query_paths, 1):
        item = extract_feature(api_url, path, args, extra_headers, cache)
        if not item.get("ok"):
            query_failures.append({"identity": identity, "path": str(path), "error": item.get("error", "")})
            print(f"  [{idx}/{len(all_query_paths)}] FAIL {identity}/{path.name}: {item.get('error')}")
            continue
        item = dict(item)
        item["identity"] = identity
        item["identity_display"] = display_name(identity)

        matches = []
        for enrolled in gallery:
            if len(item["feature"]) != len(enrolled["feature"]):
                continue
            score = cosine(item["feature"], enrolled["feature"])
            matches.append({"identity": enrolled["identity"], "score": score, "enrollment": enrolled})
        matches.sort(key=lambda m: m["score"], reverse=True)

        correct_rank = None
        correct_score = None
        for rank, match in enumerate(matches, 1):
            if match["identity"] == identity:
                correct_rank = rank
                correct_score = match["score"]
                break

        top1 = matches[0] if matches else None
        row = {
            "identity": identity,
            "identity_display": display_name(identity),
            "query": item,
            "top_matches": matches[: args.top_k],
            "correct_rank": correct_rank,
            "correct_score": correct_score,
            "top1_identity": top1["identity"] if top1 else None,
            "top1_score": top1["score"] if top1 else None,
            "top1_correct": bool(top1 and top1["identity"] == identity),
            "topk_hit": bool(correct_rank is not None and correct_rank <= args.top_k),
        }
        rows.append(row)
        rank_text = correct_rank if correct_rank is not None else "未入库/未命中"
        print(f"  [{idx}/{len(all_query_paths)}] OK rank={rank_text} top1={row['top1_identity']} {fmt(row['top1_score'])}")

        if args.save_every > 0 and idx % args.save_every == 0:
            save_cache(args.cache_json, cache, args.api_url)

    return rows, query_failures, all_query_paths


def summarize_distribution(all_people, subset):
    hist = {}
    for images in all_people.values():
        n = len(images)
        hist[n] = hist.get(n, 0) + 1
    return {
        "all_people": len(all_people),
        "all_images": sum(len(v) for v in all_people.values()),
        "subset_people": len(subset),
        "subset_images": sum(len(v) for v in subset.values()),
        "single_image_people": hist.get(1, 0),
        "multi_image_people": len(all_people) - hist.get(1, 0),
        "hist": hist,
    }


def compute_metrics(rows, query_failures, all_query_paths, top_k):
    valid = len(rows)
    total = len(all_query_paths)
    rank_hits = {}
    for k in range(1, top_k + 1):
        rank_hits[k] = sum(1 for r in rows if r["correct_rank"] is not None and r["correct_rank"] <= k)
    top1 = rank_hits.get(1, 0)
    top5 = rank_hits.get(min(5, top_k), 0)
    top10 = rank_hits.get(min(10, top_k), 0)
    ranks = [r["correct_rank"] for r in rows if r["correct_rank"] is not None]
    correct_scores = [r["correct_score"] for r in rows if r["correct_score"] is not None]
    top1_scores = [r["top1_score"] for r in rows if r["top1_score"] is not None]
    return {
        "query_total": total,
        "query_valid": valid,
        "query_failed": len(query_failures),
        "rank_hits": rank_hits,
        "top1_hits": top1,
        "top5_hits": top5,
        "top10_hits": top10,
        "top1_accuracy_valid": top1 / valid if valid else None,
        "top5_accuracy_valid": top5 / valid if valid else None,
        "top10_accuracy_valid": top10 / valid if valid else None,
        "top1_accuracy_total": top1 / total if total else None,
        "top10_accuracy_total": top10 / total if total else None,
        "median_rank": statistics.median(ranks) if ranks else None,
        "mean_rank": statistics.fmean(ranks) if ranks else None,
        "median_correct_score": statistics.median(correct_scores) if correct_scores else None,
        "median_top1_score": statistics.median(top1_scores) if top1_scores else None,
    }


def compute_person_stats(subset, enrollments, rows, query_failures, top_k):
    grouped = {name: [] for name in subset}
    failures = {name: 0 for name in subset}
    for row in rows:
        grouped[row["identity"]].append(row)
    for fail in query_failures:
        failures[fail["identity"]] = failures.get(fail["identity"], 0) + 1

    stats = []
    for identity, images in subset.items():
        person_rows = grouped.get(identity, [])
        valid = len(person_rows)
        top1 = sum(1 for r in person_rows if r["correct_rank"] == 1)
        topk = sum(1 for r in person_rows if r["correct_rank"] is not None and r["correct_rank"] <= top_k)
        correct_scores = [r["correct_score"] for r in person_rows if r["correct_score"] is not None]
        worst = max(person_rows, key=lambda r: r["correct_rank"] or 10**9) if person_rows else None
        stats.append(
            {
                "identity": identity,
                "identity_display": display_name(identity),
                "images": len(images),
                "enrolled": identity in enrollments,
                "query_valid": valid,
                "query_failed": failures.get(identity, 0),
                "top1_hits": top1,
                "topk_hits": topk,
                "top1_accuracy": top1 / valid if valid else None,
                "topk_accuracy": topk / valid if valid else None,
                "median_correct_score": statistics.median(correct_scores) if correct_scores else None,
                "worst_rank": worst["correct_rank"] if worst else None,
                "worst_query": worst["query"]["name"] if worst else "",
            }
        )
    return stats


def verdict(score, same_threshold, review_threshold):
    if score is None:
        return "无结果", "bad"
    if score >= same_threshold:
        return "建议同人", "same"
    if score >= review_threshold:
        return "需要复核", "review"
    return "相似度偏低", "diff"


def bbox_style(item):
    bbox = item.get("bbox") or []
    width = item.get("width")
    height = item.get("height")
    if not bbox or len(bbox) < 4 or not width or not height:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    left = max(0.0, min(100.0, x1 / width * 100.0))
    top = max(0.0, min(100.0, y1 / height * 100.0))
    box_w = max(0.0, min(100.0 - left, (x2 - x1) / width * 100.0))
    box_h = max(0.0, min(100.0 - top, (y2 - y1) / height * 100.0))
    return f"left:{left:.3f}%;top:{top:.3f}%;width:{box_w:.3f}%;height:{box_h:.3f}%;"


def vector_strip(vector, bins=64):
    if not vector:
        return ""
    bins = min(bins, len(vector))
    max_abs = max(abs(v) for v in vector) or 1.0
    parts = []
    for idx in range(bins):
        start = int(idx * len(vector) / bins)
        end = max(start + 1, int((idx + 1) * len(vector) / bins))
        value = sum(vector[start:end]) / (end - start)
        intensity = min(1.0, abs(value) / max_abs * 4.0)
        alpha = 0.16 + intensity * 0.84
        color = f"rgba(185, 56, 38, {alpha:.3f})" if value >= 0 else f"rgba(37, 99, 235, {alpha:.3f})"
        parts.append(f'<span title="{value:+.5f}" style="background:{color}"></span>')
    return "".join(parts)


def vector_details(item, title, compact=False):
    vector = item.get("feature") or []
    preview = ", ".join(f"{v:+.4f}" for v in vector[:8])
    full = json.dumps([round(v, 8) for v in vector], ensure_ascii=False)
    cls = " vector compact" if compact else " vector"
    return f"""
      <details class="{cls}">
        <summary>
          <span class="summary-title">{esc(title)}</span>
          <span class="dim">{len(vector)} 维</span>
          <span class="strip">{vector_strip(vector)}</span>
          <span class="preview">{esc(preview)} ...</span>
        </summary>
        <pre>{esc(full)}</pre>
      </details>
    """


def photo_block(item, label, image_mode):
    src = image_src(item["path"], image_mode)
    box = bbox_style(item)
    bbox_html = f'<span class="bbox" style="{box}"></span>' if box else ""
    score = ""
    if item.get("bbox") and len(item["bbox"]) >= 5:
        score = f"检测分 {float(item['bbox'][4]):.3f}"
    meta = f"{item.get('width') or '-'} x {item.get('height') or '-'} / {score}"
    return f"""
      <figure class="photo">
        <figcaption>{esc(label)}</figcaption>
        <div class="image-wrap">
          <img src="{src}" alt="{esc(item['name'])}">
          {bbox_html}
        </div>
        <p class="file" title="{esc(item['path'])}">{esc(item['name'])}</p>
        <p class="muted">{esc(meta)}</p>
      </figure>
    """


def match_tile(match, expected_identity, args):
    enrolled = match["enrollment"]
    is_correct = match["identity"] == expected_identity
    label = "正确身份" if is_correct else "其他身份"
    cls = "hit" if is_correct else "miss"
    src = image_src(enrolled["path"], args.image_mode)
    box = bbox_style(enrolled)
    bbox_html = f'<span class="bbox" style="{box}"></span>' if box else ""
    return f"""
      <div class="match-tile {cls}">
        <div class="tile-img">
          <img src="{src}" alt="{esc(enrolled['name'])}">
          {bbox_html}
        </div>
        <div class="tile-name">{esc(display_name(match['identity']))}</div>
        <div class="tile-score">{fmt(match['score'])}</div>
        <div class="tile-label">{esc(label)}</div>
        {vector_details(enrolled, "底库向量", compact=True)}
      </div>
    """


def case_card(row, args, title_prefix):
    query = row["query"]
    rank = row["correct_rank"]
    top1_score = row["top1_score"]
    label, cls = verdict(row["correct_score"], args.same_threshold, args.review_threshold)
    if rank == 1:
        result = "Top1 命中"
        result_cls = "same"
    elif rank is not None and rank <= args.top_k:
        result = f"Top{rank} 命中"
        result_cls = "review"
    else:
        result = f"Top{args.top_k} 未命中"
        result_cls = "bad"
    tiles = "\n".join(match_tile(m, row["identity"], args) for m in row["top_matches"])
    return f"""
    <section class="case-card {result_cls}">
      <div class="case-head">
        <div>
          <div class="case-prefix">{esc(title_prefix)}</div>
          <h3>{esc(display_name(row['identity']))} / {esc(query['name'])}</h3>
          <p class="muted">真实身份排名：{rank if rank is not None else 'TopK 外'}，正确身份相似度：{fmt(row['correct_score'])}，Top1 相似度：{fmt(top1_score)}</p>
        </div>
        <div class="badge">{esc(result)} · {esc(label)}</div>
      </div>
      <div class="case-grid">
        <div>
          {photo_block(query, "查询图", args.image_mode)}
          {vector_details(query, "查询图向量")}
        </div>
        <div class="top-grid">{tiles}</div>
      </div>
    </section>
    """


def limit_cases(cases, limit):
    if limit <= 0:
        return cases
    return cases[:limit]


def build_case_sections(rows, args):
    if args.detail_mode == "none":
        return '<section class="panel"><p>已关闭可视化案例展示。</p></section>'

    if args.detail_mode == "all":
        all_cases = sorted(rows, key=lambda r: (r["identity"], r["query"]["name"]))
        all_cases = limit_cases(all_cases, args.max_case_cards)
        cards = "\n".join(case_card(r, args, "全部查询明细") for r in all_cases)
        if not cards:
            cards = '<section class="panel">无可展示案例。</section>'
        return f"<h2>全部查询 Top{args.top_k} 明细</h2>{cards}"

    misses = [r for r in rows if r["correct_rank"] is None or r["correct_rank"] > args.top_k]
    misses.sort(key=lambda r: r["top1_score"] or -1, reverse=True)

    top1_wrong_hit = [r for r in rows if r["correct_rank"] is not None and 1 < r["correct_rank"] <= args.top_k]
    top1_wrong_hit.sort(key=lambda r: (r["top1_score"] or 0) - (r["correct_score"] or 0), reverse=True)

    weak_success = [r for r in rows if r["correct_rank"] == 1]
    weak_success.sort(key=lambda r: r["correct_score"] or 0)

    rng = random.Random(args.seed)
    success = [r for r in rows if r["correct_rank"] == 1]
    rng.shuffle(success)

    sections = []
    for title, cases, prefix in [
        (f"Top{args.top_k} 未命中案例", misses, "重点错误"),
        (f"Top1 错误但 Top{args.top_k} 命中案例", top1_wrong_hit, "可复核案例"),
        ("Top1 命中但相似度较低样本", weak_success, "低置信样本"),
        ("Top1 命中样本抽样", success[: args.success_samples], "正确样本"),
    ]:
        selected = limit_cases(cases, args.max_case_cards)
        cards = "\n".join(case_card(r, args, prefix) for r in selected)
        if not cards:
            cards = '<section class="panel"><p>无。</p></section>'
        sections.append(f"<h2>{esc(title)}</h2>{cards}")
    return "\n".join(sections)


def style_block():
    return """
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #d9dee7;
      --brand: #0f766e;
      --good: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
      --blue: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      line-height: 1.5;
    }
    header { padding: 28px 34px 22px; background: #111827; color: #fff; }
    header h1 { margin: 0 0 10px; font-size: 28px; letter-spacing: 0; }
    header p { margin: 3px 0; color: #d1d5db; }
    main { padding: 24px 34px 42px; }
    h2 { margin: 28px 0 12px; font-size: 22px; }
    h3 { margin: 2px 0 4px; font-size: 18px; }
    .notice { border-left: 4px solid var(--brand); background: #ecfdf5; padding: 12px 14px; margin-bottom: 18px; }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }
    .metric, .panel, .case-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .metric { padding: 12px 14px; }
    .metric .k { color: var(--muted); font-size: 13px; }
    .metric .v { font-size: 22px; font-weight: 800; margin-top: 2px; }
    .panel { padding: 14px; margin-bottom: 16px; }
    table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }
    th { background: #f3f4f6; font-weight: 800; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .muted { color: var(--muted); font-size: 13px; margin: 0; }
    .case-card { padding: 16px; margin-bottom: 16px; }
    .case-head { display: flex; justify-content: space-between; gap: 18px; border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 14px; }
    .case-prefix { color: var(--muted); font-weight: 800; font-size: 13px; }
    .badge { align-self: flex-start; color: #fff; background: var(--muted); border-radius: 999px; padding: 5px 10px; font-size: 13px; white-space: nowrap; }
    .case-card.same .badge { background: var(--good); }
    .case-card.review .badge { background: var(--warn); }
    .case-card.bad .badge { background: var(--bad); }
    .case-grid { display: grid; grid-template-columns: minmax(220px, 300px) 1fr; gap: 18px; align-items: start; }
    .photo { margin: 0; }
    .photo figcaption { font-weight: 800; margin-bottom: 8px; }
    .image-wrap, .tile-img { position: relative; background: #e5e7eb; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
    .image-wrap { display: inline-block; max-width: 300px; }
    .image-wrap img { display: block; max-width: 100%; height: auto; }
    .bbox { position: absolute; display: block; border: 2px solid #22c55e; box-shadow: 0 0 0 1px rgba(255,255,255,.8); pointer-events: none; }
    .file { margin: 8px 0 2px; font-weight: 800; word-break: break-all; }
    .top-grid { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 12px; }
    .match-tile { border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #fff; }
    .match-tile.hit { border-color: #16a34a; box-shadow: inset 0 0 0 1px #16a34a; }
    .match-tile.miss { opacity: .92; }
    .tile-img { aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center; }
    .tile-img img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .tile-name { margin-top: 6px; font-weight: 800; font-size: 13px; min-height: 38px; }
    .tile-score { font-size: 18px; font-weight: 900; font-variant-numeric: tabular-nums; }
    .tile-label { color: var(--muted); font-size: 12px; }
    .vector { border: 1px solid var(--line); border-radius: 6px; margin-top: 10px; overflow: hidden; }
    .vector.compact { margin-top: 8px; }
    .vector summary { cursor: pointer; padding: 9px 10px; background: #f9fafb; list-style-position: inside; }
    .vector.compact summary { padding: 6px; }
    .summary-title { font-weight: 800; margin-right: 8px; }
    .dim, .preview { color: var(--muted); font-size: 12px; }
    .strip { display: inline-grid; grid-template-columns: repeat(64, 4px); gap: 1px; vertical-align: middle; margin: 0 8px; }
    .strip span { width: 4px; height: 13px; border-radius: 1px; display: block; }
    pre { white-space: pre-wrap; word-break: break-all; margin: 0; padding: 10px; background: #111827; color: #e5e7eb; max-height: 240px; overflow: auto; font-size: 12px; }
    .bar { display: inline-block; height: 9px; background: var(--brand); border-radius: 99px; min-width: 2px; }
    @media (max-width: 1100px) {
      main { padding: 16px; }
      header { padding: 22px 16px; }
      .metrics { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .case-head { flex-direction: column; }
      .case-grid { grid-template-columns: 1fr; }
      .top-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
    """


def hist_rows(hist):
    buckets = [
        ("1 张", sum(v for k, v in hist.items() if k == 1)),
        ("2 张", sum(v for k, v in hist.items() if k == 2)),
        ("3 张", sum(v for k, v in hist.items() if k == 3)),
        ("4 张", sum(v for k, v in hist.items() if k == 4)),
        ("5 张", sum(v for k, v in hist.items() if k == 5)),
        ("6-10 张", sum(v for k, v in hist.items() if 6 <= k <= 10)),
        ("11-19 张", sum(v for k, v in hist.items() if 11 <= k <= 19)),
        (">=20 张", sum(v for k, v in hist.items() if k >= 20)),
    ]
    max_count = max((count for _, count in buckets), default=1)
    rows = []
    for label, count in buckets:
        width = count / max_count * 100 if max_count else 0
        rows.append(f"<tr><td>{esc(label)}</td><td class='num'>{count}</td><td><span class='bar' style='width:{width:.1f}%'></span></td></tr>")
    return "\n".join(rows)


def cmc_rows(metrics, top_k):
    rows = []
    valid = metrics["query_valid"]
    for k in range(1, top_k + 1):
        hit = metrics["rank_hits"].get(k, 0)
        rows.append(f"<tr><td>Rank-{k}</td><td class='num'>{hit}</td><td class='num'>{pct(hit, valid)}</td></tr>")
    return "\n".join(rows)


def person_rows(person_stats, top_k):
    rows = []
    for s in sorted(person_stats, key=lambda x: (x["topk_accuracy"] if x["topk_accuracy"] is not None else -1, x["top1_accuracy"] if x["top1_accuracy"] is not None else -1, x["identity"])):
        rows.append(
            "<tr>"
            f"<td>{esc(s['identity_display'])}</td>"
            f"<td class='num'>{s['images']}</td>"
            f"<td class='num'>{s['query_valid']}</td>"
            f"<td class='num'>{s['query_failed']}</td>"
            f"<td class='num'>{pct(s['top1_hits'], s['query_valid'])}</td>"
            f"<td class='num'>{pct(s['topk_hits'], s['query_valid'])}</td>"
            f"<td class='num'>{fmt(s['median_correct_score'])}</td>"
            f"<td>{esc(s['worst_query'])}</td>"
            f"<td class='num'>{s['worst_rank'] if s['worst_rank'] is not None else '-'}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def failure_rows(failures, limit=200):
    rows = []
    for item in failures[:limit]:
        rows.append(f"<tr><td>{esc(display_name(item.get('identity', '')))}</td><td>{esc(item.get('path', ''))}</td><td>{esc(item.get('error', ''))}</td></tr>")
    if not rows:
        return '<tr><td colspan="3">无</td></tr>'
    if len(failures) > limit:
        rows.append(f'<tr><td colspan="3">仅展示前 {limit} 条，共 {len(failures)} 条。</td></tr>')
    return "\n".join(rows)


def make_html(args, distribution, subset, enrollments, enrollment_failures, rows, query_failures, metrics, person_stats, generated_at):
    subset_top = sorted(((name, len(images)) for name, images in subset.items()), key=lambda x: (-x[1], x[0]))[:12]
    subset_top_rows = "\n".join(f"<tr><td>{esc(display_name(name))}</td><td class='num'>{count}</td></tr>" for name, count in subset_top)
    case_sections = build_case_sections(rows, args)
    valid = metrics["query_valid"]
    total = metrics["query_total"]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(args.title)}</title>
  <style>{style_block()}</style>
</head>
<body>
  <header>
    <h1>{esc(args.title)}</h1>
    <p>生成时间：{esc(generated_at)}</p>
    <p>接口地址：{esc(args.api_url)}</p>
    <p>测试口径：闭集 1:N 识别，每人 1 张底库注册照，其余图片作为查询图，按相似度排序取 Top{args.top_k}。</p>
  </header>
  <main>
    <div class="notice">
      本报告回答的问题是：当这 {len(enrollments)} 个人已经入库时，新来一张属于这批人的人脸图，系统能否在 Top{args.top_k} 里找回正确身份。
      这是闭集识别测试，不包含库外陌生人拒识能力。
    </div>

    <section class="metrics">
      <div class="metric"><div class="k">LFW 全量人物</div><div class="v">{distribution['all_people']}</div></div>
      <div class="metric"><div class="k">LFW 全量图片</div><div class="v">{distribution['all_images']}</div></div>
      <div class="metric"><div class="k">本次子集人物</div><div class="v">{len(subset)}</div></div>
      <div class="metric"><div class="k">本次子集图片</div><div class="v">{distribution['subset_images']}</div></div>
      <div class="metric"><div class="k">成功入库人数</div><div class="v">{len(enrollments)}</div></div>
      <div class="metric"><div class="k">查询图片</div><div class="v">{total}</div></div>
      <div class="metric"><div class="k">有效查询</div><div class="v">{valid}</div></div>
      <div class="metric"><div class="k">查询失败</div><div class="v">{metrics['query_failed']}</div></div>
      <div class="metric"><div class="k">Top1 准确率</div><div class="v">{pct(metrics['top1_hits'], valid)}</div></div>
      <div class="metric"><div class="k">Top{args.top_k} 准确率</div><div class="v">{pct(metrics['top10_hits'], valid)}</div></div>
    </section>

    <h2>数据集概览</h2>
    <div class="panel">
      <p>LFW funneled 全量共 {distribution['all_images']} 张图片、{distribution['all_people']} 个人；其中只有 1 张图的人有 {distribution['single_image_people']} 个，有 2 张及以上的人有 {distribution['multi_image_people']} 个。</p>
      <p>本次使用每人至少 {args.min_images_per_person} 张图的子集：{len(subset)} 个人、{distribution['subset_images']} 张图片。</p>
    </div>
    <table>
      <thead><tr><th>每人图片数</th><th class="num">人数</th><th>分布</th></tr></thead>
      <tbody>{hist_rows(distribution['hist'])}</tbody>
    </table>

    <h2>子集图片最多的人物</h2>
    <table>
      <thead><tr><th>人物</th><th class="num">图片数</th></tr></thead>
      <tbody>{subset_top_rows}</tbody>
    </table>

    <h2>CMC / Rank 命中率</h2>
    <table>
      <thead><tr><th>指标</th><th class="num">命中数</th><th class="num">有效查询口径</th></tr></thead>
      <tbody>{cmc_rows(metrics, args.top_k)}</tbody>
    </table>

    <h2>按人统计</h2>
    <table>
      <thead>
        <tr>
          <th>人物</th><th class="num">图片数</th><th class="num">有效查询</th><th class="num">失败</th>
          <th class="num">Top1</th><th class="num">Top{args.top_k}</th><th class="num">正确身份中位相似度</th>
          <th>最差样本</th><th class="num">最差排名</th>
        </tr>
      </thead>
      <tbody>{person_rows(person_stats, args.top_k)}</tbody>
    </table>

    {case_sections}

    <h2>失败明细</h2>
    <table>
      <thead><tr><th>人物</th><th>图片路径</th><th>原因</th></tr></thead>
      <tbody>{failure_rows(enrollment_failures + query_failures)}</tbody>
    </table>
  </main>
</body>
</html>
"""


def write_json(path, args, distribution, subset, enrollments, enrollment_failures, rows, query_failures, metrics, person_stats, generated_at):
    def strip_feature(item):
        return {k: v for k, v in item.items() if k != "feature"}

    payload = {
        "title": args.title,
        "generated_at": generated_at,
        "api_url": args.api_url,
        "lfw_dir": args.lfw_dir,
        "min_images_per_person": args.min_images_per_person,
        "top_k": args.top_k,
        "distribution": distribution,
        "subset_people": [{"identity": name, "images": len(images)} for name, images in subset.items()],
        "metrics": metrics,
        "person_stats": person_stats,
        "enrollments": {identity: strip_feature(item) for identity, item in enrollments.items()},
        "enrollment_failures": enrollment_failures,
        "query_failures": query_failures,
        "queries": [
            {
                "identity": row["identity"],
                "query": strip_feature(row["query"]),
                "correct_rank": row["correct_rank"],
                "correct_score": row["correct_score"],
                "top1_identity": row["top1_identity"],
                "top1_score": row["top1_score"],
                "topk_hit": row["topk_hit"],
                "top_matches": [
                    {
                        "identity": m["identity"],
                        "score": m["score"],
                        "enrollment_path": m["enrollment"]["path"],
                    }
                    for m in row["top_matches"]
                ],
            }
            for row in rows
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    args.api_url = normalize_api_url(args.api_url)
    extra_headers = parse_headers(args.header)
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_html = Path(args.out_html)
    out_json = Path(args.out_json) if args.out_json else out_html.with_suffix(".json")

    all_people, subset = scan_lfw(args.lfw_dir, args.min_images_per_person)
    distribution = summarize_distribution(all_people, subset)
    if not subset:
        raise RuntimeError("筛选后没有可测试身份")

    print(f"[数据] LFW 全量 {distribution['all_people']} 人 / {distribution['all_images']} 图")
    print(f"[数据] 子集 {len(subset)} 人 / {distribution['subset_images']} 图")

    cache = load_cache(args.cache_json, args.no_cache)
    enrollments, enrollment_failures = build_enrollment(subset, args.api_url, args, extra_headers, cache)
    if not enrollments:
        raise RuntimeError("没有任何身份成功入库")

    rows, query_failures, all_query_paths = evaluate_queries(subset, enrollments, args.api_url, args, extra_headers, cache)
    save_cache(args.cache_json, cache, args.api_url)

    metrics = compute_metrics(rows, query_failures, all_query_paths, args.top_k)
    person_stats = compute_person_stats(subset, enrollments, rows, query_failures, args.top_k)

    html_text = make_html(args, distribution, subset, enrollments, enrollment_failures, rows, query_failures, metrics, person_stats, generated_at)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")
    write_json(out_json, args, distribution, subset, enrollments, enrollment_failures, rows, query_failures, metrics, person_stats, generated_at)

    print("[完成] 报告已生成")
    print(f"HTML: {out_html}")
    print(f"JSON: {out_json}")
    print(f"Cache: {args.cache_json}")
    print(f"Top1: {pct(metrics['top1_hits'], metrics['query_valid'])}")
    print(f"Top{args.top_k}: {pct(metrics['top10_hits'], metrics['query_valid'])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
