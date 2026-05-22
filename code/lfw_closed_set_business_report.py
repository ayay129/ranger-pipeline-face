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
    parser.add_argument("--max-case-cards", type=int, default=0, help="每类最多展示多少个案例，0 表示不限制")
    parser.add_argument("--success-samples", type=int, default=0, help="Top1 正确样本抽样展示数量，0 表示不抽样")
    parser.add_argument(
        "--image-mode",
        choices=("relative", "file", "embed"),
        default="relative",
        help="图片引用方式：relative 适合项目根目录 http.server 且文件小；embed 可独立分享但文件很大；file 只适合本机直接打开",
    )
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
    if image_mode == "relative":
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.as_posix()
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
        correct_match = None
        for rank, match in enumerate(matches, 1):
            if match["identity"] == identity:
                correct_rank = rank
                correct_score = match["score"]
                correct_match = match
                break

        top1 = matches[0] if matches else None
        row = {
            "identity": identity,
            "identity_display": display_name(identity),
            "query": item,
            "top_matches": matches[: args.top_k],
            "correct_rank": correct_rank,
            "correct_score": correct_score,
            "correct_match": correct_match,
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


def compact_photo(item, label, identity, score, image_mode, tone="neutral"):
    src = image_src(item["path"], image_mode)
    box = bbox_style(item)
    bbox_html = f'<span class="bbox" style="{box}"></span>' if box else ""
    score_html = f"<div class='focus-score'>相似度 {fmt(score)}</div>" if score is not None else ""
    return f"""
      <div class="focus-photo {tone}">
        <div class="focus-label">{esc(label)}</div>
        <div class="focus-img">
          <img src="{src}" alt="{esc(item['name'])}">
          {bbox_html}
        </div>
        <div class="focus-name">{esc(display_name(identity))}</div>
        <div class="focus-file" title="{esc(item['path'])}">{esc(item['name'])}</div>
        {score_html}
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
    top1_match = row["top_matches"][0] if row["top_matches"] else None
    correct_match = row.get("correct_match")
    tiles = "\n".join(match_tile(m, row["identity"], args) for m in row["top_matches"])
    top1_block = ""
    if top1_match:
        top1_tone = "hit" if top1_match["identity"] == row["identity"] else "miss"
        top1_block = compact_photo(
            top1_match["enrollment"],
            "Top1 命中结果",
            top1_match["identity"],
            top1_match["score"],
            args.image_mode,
            top1_tone,
        )
    correct_block = ""
    if correct_match:
        correct_block = compact_photo(
            correct_match["enrollment"],
            "正确身份底库",
            correct_match["identity"],
            correct_match["score"],
            args.image_mode,
            "hit",
        )
    elif top1_match:
        correct_block = """
          <div class="focus-photo empty">
            <div class="focus-label">正确身份底库</div>
            <div class="empty-box">正确身份未进入当前底库匹配结果</div>
          </div>
        """
    correct_rank_text = rank if rank is not None else f"Top{args.top_k} 外"
    top1_identity = display_name(row["top1_identity"]) if row["top1_identity"] else "-"
    return f"""
    <details class="case-card {result_cls}">
      <summary class="case-summary">
        <span class="badge">{esc(result)}</span>
        <strong>{esc(display_name(row['identity']))} / {esc(query['name'])}</strong>
        <span>正确排名：{esc(correct_rank_text)}</span>
        <span>Top1：{esc(top1_identity)} {fmt(top1_score)}</span>
        <span>正确相似度：{fmt(row['correct_score'])}</span>
      </summary>
      <div class="case-body">
        <div class="case-prefix">{esc(title_prefix)} · {esc(label)}</div>
        <div class="focus-grid">
          {compact_photo(query, "查询图", row["identity"], None, args.image_mode, "query")}
          {top1_block}
          {correct_block}
        </div>
        <details class="nested-detail">
          <summary>查看完整 Top{args.top_k} 候选</summary>
          <div class="top-grid">{tiles}</div>
        </details>
        <details class="nested-detail">
          <summary>查看向量缩略与完整 512 维</summary>
          {vector_details(query, "查询图向量")}
          {vector_details(top1_match["enrollment"], "Top1 底库向量") if top1_match else ""}
          {vector_details(correct_match["enrollment"], "正确身份底库向量") if correct_match and (not top1_match or correct_match["identity"] != top1_match["identity"]) else ""}
        </details>
      </div>
    </details>
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
    .case-card { margin-bottom: 12px; overflow: hidden; }
    .case-card[open] { margin-bottom: 18px; }
    .case-summary {
      cursor: pointer;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px 14px;
      padding: 13px 16px;
      background: #fff;
      border-bottom: 1px solid transparent;
      list-style-position: inside;
    }
    .case-card[open] .case-summary { border-bottom-color: var(--line); background: #fbfcfd; }
    .case-summary strong { font-size: 16px; }
    .case-summary span:not(.badge) { color: var(--muted); font-size: 13px; }
    .case-body { padding: 16px; }
    .case-head { display: flex; justify-content: space-between; gap: 18px; border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 14px; }
    .case-prefix { color: var(--muted); font-weight: 800; font-size: 13px; }
    .badge { align-self: flex-start; color: #fff; background: var(--muted); border-radius: 999px; padding: 5px 10px; font-size: 13px; white-space: nowrap; }
    .case-card.same .badge { background: var(--good); }
    .case-card.review .badge { background: var(--warn); }
    .case-card.bad .badge { background: var(--bad); }
    .case-grid { display: grid; grid-template-columns: minmax(220px, 300px) 1fr; gap: 18px; align-items: start; }
    .focus-grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 16px; margin-top: 12px; }
    .focus-photo { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 10px; }
    .focus-photo.hit { border-color: #16a34a; box-shadow: inset 0 0 0 1px #16a34a; }
    .focus-photo.miss { border-color: #f97316; box-shadow: inset 0 0 0 1px #f97316; }
    .focus-photo.query { border-color: #94a3b8; }
    .focus-label { font-weight: 900; margin-bottom: 8px; }
    .focus-img { position: relative; background: #e5e7eb; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; min-height: 240px; display: flex; align-items: center; justify-content: center; }
    .focus-img img { width: 100%; max-height: 420px; object-fit: contain; display: block; }
    .focus-name { margin-top: 8px; font-weight: 900; font-size: 16px; }
    .focus-file { color: var(--muted); font-size: 13px; word-break: break-all; }
    .focus-score { margin-top: 4px; font-size: 22px; font-weight: 900; font-variant-numeric: tabular-nums; }
    .empty-box { min-height: 240px; border: 1px dashed var(--line); border-radius: 6px; display: flex; align-items: center; justify-content: center; color: var(--muted); text-align: center; padding: 18px; }
    .nested-detail { border: 1px solid var(--line); border-radius: 8px; margin-top: 14px; overflow: hidden; }
    .nested-detail > summary { cursor: pointer; padding: 10px 12px; background: #f9fafb; font-weight: 900; }
    .nested-detail .top-grid { padding: 12px; }
    .nested-detail > .vector { margin: 12px; }
    .photo { margin: 0; }
    .photo figcaption { font-weight: 800; margin-bottom: 8px; }
    .image-wrap, .tile-img { position: relative; background: #e5e7eb; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
    .image-wrap { display: inline-block; max-width: 300px; }
    .image-wrap img { display: block; max-width: 100%; height: auto; }
    .bbox { display: none !important; }
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
      .case-summary { align-items: flex-start; }
      .case-grid { grid-template-columns: 1fr; }
      .focus-grid { grid-template-columns: 1fr; }
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


def image_payload(item, args, identity=None, score=None):
    feature = item.get("feature") or []
    return {
        "src": image_src(item["path"], args.image_mode),
        "name": item.get("name", Path(item.get("path", "")).name),
        "path": item.get("path", ""),
        "identity": identity,
        "identityDisplay": display_name(identity) if identity else "",
        "score": score,
        "bboxStyle": bbox_style(item),
        "width": item.get("width"),
        "height": item.get("height"),
        "featureDim": len(feature),
        "featurePreview": [round(v, 6) for v in feature[:16]],
    }


def case_payload(row, args):
    top1 = row["top_matches"][0] if row["top_matches"] else None
    correct = row.get("correct_match")
    return {
        "identity": row["identity"],
        "identityDisplay": display_name(row["identity"]),
        "queryName": row["query"]["name"],
        "status": "Top1正确" if row["correct_rank"] == 1 else (f"Top{row['correct_rank']}可召回" if row["correct_rank"] and row["correct_rank"] <= args.top_k else f"Top{args.top_k}未命中"),
        "correctRank": row["correct_rank"],
        "correctScore": row["correct_score"],
        "top1Identity": row["top1_identity"],
        "top1IdentityDisplay": display_name(row["top1_identity"]) if row["top1_identity"] else "",
        "top1Score": row["top1_score"],
        "gap": (row["top1_score"] - row["correct_score"]) if row["top1_score"] is not None and row["correct_score"] is not None else None,
        "query": image_payload(row["query"], args, row["identity"], None),
        "top1": image_payload(top1["enrollment"], args, top1["identity"], top1["score"]) if top1 else None,
        "correct": image_payload(correct["enrollment"], args, correct["identity"], correct["score"]) if correct else None,
        "topMatches": [
            {
                "rank": idx + 1,
                "identity": match["identity"],
                "identityDisplay": display_name(match["identity"]),
                "score": match["score"],
                "isCorrect": match["identity"] == row["identity"],
                "image": image_payload(match["enrollment"], args, match["identity"], match["score"]),
            }
            for idx, match in enumerate(row["top_matches"])
        ],
    }


def limited_cases(cases, limit):
    if limit <= 0:
        return cases
    return cases[:limit]


def build_dashboard_data(args, distribution, subset, enrollments, enrollment_failures, rows, query_failures, metrics, person_stats, generated_at):
    success = [r for r in rows if r["correct_rank"] == 1]
    success.sort(key=lambda r: r["correct_score"] or 0)

    top5 = [r for r in rows if r["correct_rank"] is not None and r["correct_rank"] <= min(5, args.top_k)]
    top5.sort(key=lambda r: (r["correct_rank"], r["correct_score"] or 0))

    top10 = [r for r in rows if r["correct_rank"] is not None and r["correct_rank"] <= args.top_k]
    top10.sort(key=lambda r: (r["correct_rank"], r["correct_score"] or 0))

    high_risk = [r for r in rows if r["correct_rank"] != 1]
    high_risk.sort(key=lambda r: r["top1_score"] or -1, reverse=True)

    miss = [r for r in rows if r["correct_rank"] is None or r["correct_rank"] > args.top_k]
    miss.sort(key=lambda r: r["top1_score"] or -1, reverse=True)

    case_limit = args.max_case_cards
    success_limit = args.success_samples if args.success_samples > 0 else case_limit
    if case_limit > 0:
        success_limit = min(success_limit, case_limit)

    return {
        "title": args.title,
        "generatedAt": generated_at,
        "apiUrl": args.api_url,
        "topK": args.top_k,
        "sameThreshold": args.same_threshold,
        "reviewThreshold": args.review_threshold,
        "imageMode": args.image_mode,
        "distribution": distribution,
        "metrics": metrics,
        "subsetPeople": len(subset),
        "subsetImages": distribution["subset_images"],
        "enrollmentPeople": len(enrollments),
        "views": {
            "success": [case_payload(r, args) for r in limited_cases(success, success_limit)],
            "top5": [case_payload(r, args) for r in limited_cases(top5, case_limit)],
            "top10": [case_payload(r, args) for r in limited_cases(top10, case_limit)],
            "risk": [case_payload(r, args) for r in limited_cases(high_risk, case_limit)],
            "miss": [case_payload(r, args) for r in limited_cases(miss, case_limit)],
        },
        "counts": {
            "success": len(success),
            "top5": len(top5),
            "top10": len(top10),
            "risk": len(high_risk),
            "miss": len(miss),
        },
        "personStats": person_stats,
        "failures": enrollment_failures + query_failures,
    }


def dashboard_style():
    return """
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --line: #d8dee8;
      --text: #172033;
      --muted: #687385;
      --brand: #0f766e;
      --good: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
      --shadow: 0 10px 30px rgba(15, 23, 42, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      line-height: 1.45;
      overflow: hidden;
    }
    .app { height: 100vh; display: grid; grid-template-columns: 240px 1fr; grid-template-rows: 74px 1fr; }
    .topbar {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 12px 20px;
      background: #111827;
      color: #fff;
    }
    .topbar h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
    .topbar .sub { color: #cbd5e1; font-size: 12px; margin-top: 2px; }
    .top-metrics { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .top-pill { background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.16); border-radius: 8px; padding: 6px 9px; min-width: 92px; }
    .top-pill .k { color: #cbd5e1; font-size: 11px; }
    .top-pill .v { font-weight: 900; font-size: 16px; }
    .sidebar {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      overflow: auto;
    }
    .nav-btn {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      border: 1px solid transparent;
      background: transparent;
      color: var(--text);
      border-radius: 8px;
      padding: 10px 10px;
      margin-bottom: 6px;
      cursor: pointer;
      text-align: left;
      font-weight: 800;
    }
    .nav-btn:hover { background: #f1f5f9; }
    .nav-btn.active { background: #e7f6f3; border-color: #99d5ca; color: #075e56; }
    .nav-count { color: var(--muted); font-size: 12px; font-weight: 700; }
    .content { overflow: hidden; padding: 16px; }
    .view { height: 100%; display: none; }
    .view.active { display: block; }
    .overview-grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin-bottom: 14px; }
    .metric-card, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
    }
    .metric-card { padding: 14px; }
    .metric-card .k { color: var(--muted); font-size: 13px; }
    .metric-card .v { margin-top: 4px; font-size: 26px; font-weight: 950; }
    .panel { padding: 14px; margin-bottom: 14px; }
    .workspace {
      height: 100%;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 14px;
      min-height: 0;
    }
    .case-list, .detail-pane {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
      overflow: hidden;
      min-height: 0;
    }
    .case-list { display: flex; flex-direction: column; }
    .list-head { padding: 12px; border-bottom: 1px solid var(--line); }
    .list-head h2 { margin: 0; font-size: 18px; }
    .list-head input { width: 100%; margin-top: 10px; padding: 9px 10px; border: 1px solid var(--line); border-radius: 8px; }
    .list-items { overflow: auto; padding: 8px; }
    .case-item { width: 100%; border: 1px solid transparent; background: #fff; border-radius: 8px; padding: 10px; margin-bottom: 8px; cursor: pointer; text-align: left; }
    .case-item:hover { background: #f8fafc; }
    .case-item.active { border-color: #0f766e; background: #ecfdf5; }
    .case-title { font-weight: 900; margin-bottom: 4px; }
    .case-meta { color: var(--muted); font-size: 12px; }
    .detail-pane { overflow: auto; padding: 16px; }
    .result-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr)); gap: 12px; }
    .mini-card { border: 1px solid var(--line); border-radius: 10px; background: #fff; overflow: hidden; }
    .mini-card.good { border-color: #16a34a; }
    .mini-card.warn { border-color: #f97316; }
    .mini-card.bad { border-color: #dc2626; }
    .mini-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; padding: 10px 12px; border-bottom: 1px solid var(--line); }
    .mini-title { font-weight: 950; word-break: break-all; }
    .mini-meta { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .mini-badge { color: #fff; border-radius: 999px; padding: 4px 8px; font-size: 12px; font-weight: 900; white-space: nowrap; }
    .mini-badge.good { background: var(--good); }
    .mini-badge.warn { background: var(--warn); }
    .mini-badge.bad { background: var(--bad); }
    .mini-compare { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 10px; }
    .mini-face { min-width: 0; }
    .mini-label { font-size: 12px; font-weight: 900; margin-bottom: 5px; }
    .mini-img { aspect-ratio: 1 / 1; border: 1px solid var(--line); border-radius: 8px; background: #e5e7eb; overflow: hidden; display: flex; align-items: center; justify-content: center; }
    .mini-img img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .mini-name { margin-top: 5px; font-size: 12px; font-weight: 900; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .mini-score { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
    .mini-actions { padding: 0 10px 10px; }
    .mini-actions details { border: 1px solid var(--line); border-radius: 8px; margin-top: 8px; overflow: hidden; }
    .mini-actions summary { cursor: pointer; padding: 8px 9px; background: #f8fafc; font-weight: 900; font-size: 13px; }
    .mini-top10 { display: grid; grid-template-columns: repeat(5, minmax(70px, 1fr)); gap: 8px; padding: 8px; }
    .mini-thumb { border: 1px solid var(--line); border-radius: 7px; padding: 5px; }
    .mini-thumb.correct { border-color: #16a34a; }
    .mini-thumb img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #e5e7eb; border-radius: 5px; display: block; }
    .mini-thumb div { font-size: 11px; margin-top: 3px; }
    .detail-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 14px; }
    .detail-head h2 { margin: 0 0 5px; font-size: 22px; }
    .badge { color: #fff; border-radius: 999px; padding: 6px 10px; font-size: 13px; font-weight: 900; white-space: nowrap; }
    .badge.good { background: var(--good); }
    .badge.warn { background: var(--warn); }
    .badge.bad { background: var(--bad); }
    .compare-grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 14px; }
    .face-card { border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #fff; }
    .face-card.good { border-color: #16a34a; box-shadow: inset 0 0 0 1px #16a34a; }
    .face-card.warn { border-color: #f97316; box-shadow: inset 0 0 0 1px #f97316; }
    .face-label { font-weight: 950; margin-bottom: 8px; }
    .img-box { position: relative; min-height: 280px; background: #e5e7eb; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; }
    .img-box img { width: 100%; max-height: 460px; object-fit: contain; display: block; }
    .bbox { display: none !important; }
    .face-name { margin-top: 8px; font-size: 16px; font-weight: 950; }
    .face-file { color: var(--muted); font-size: 12px; word-break: break-all; }
    .face-score { margin-top: 5px; font-size: 24px; font-weight: 950; font-variant-numeric: tabular-nums; }
    .strip-panel { margin-top: 14px; border: 1px solid var(--line); border-radius: 10px; padding: 12px; background: #fff; }
    .strip-panel h3 { margin: 0 0 10px; }
    .top-strip { display: grid; grid-template-columns: repeat(10, minmax(90px, 1fr)); gap: 10px; }
    .thumb { border: 1px solid var(--line); border-radius: 8px; padding: 7px; background: #fff; }
    .thumb.correct { border-color: #16a34a; box-shadow: inset 0 0 0 1px #16a34a; }
    .thumb-img { aspect-ratio: 1/1; background: #e5e7eb; border-radius: 6px; overflow: hidden; display: flex; align-items: center; justify-content: center; }
    .thumb-img img { width: 100%; height: 100%; object-fit: contain; }
    .thumb-rank { font-size: 12px; color: var(--muted); margin-top: 5px; }
    .thumb-name { font-size: 12px; font-weight: 900; min-height: 34px; }
    .thumb-score { font-size: 15px; font-weight: 950; }
    details.tech { margin-top: 12px; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    details.tech > summary { cursor: pointer; padding: 10px; background: #f8fafc; font-weight: 900; }
    pre { margin: 0; padding: 10px; white-space: pre-wrap; word-break: break-all; background: #111827; color: #e5e7eb; max-height: 240px; overflow: auto; font-size: 12px; }
    table { width: 100%; border-collapse: collapse; background: #fff; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; }
    th { background: #f8fafc; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    @media (max-width: 1180px) {
      body { overflow: auto; }
      .app { height: auto; min-height: 100vh; grid-template-columns: 1fr; grid-template-rows: auto auto 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); display: flex; gap: 8px; overflow-x: auto; }
      .nav-btn { min-width: 150px; }
      .content { overflow: visible; }
      .workspace { grid-template-columns: 1fr; }
      .result-grid { grid-template-columns: 1fr; }
      .compare-grid { grid-template-columns: 1fr; }
      .top-strip { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
    """


def dashboard_script():
    return """
    const DATA = JSON.parse(document.getElementById('report-data').textContent);
    const $ = (sel, root = document) => root.querySelector(sel);
    const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
    const state = { view: 'overview', selectedPerson: { success: '', top5: '', top10: '', risk: '', miss: '' }, filters: {} };
    const fmt = (v, d = 4) => v === null || v === undefined ? '-' : Number(v).toFixed(d);
    const pct = (v) => v === null || v === undefined ? '-' : (Number(v) * 100).toFixed(2) + '%';
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));

    function nav() {
      $$('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === state.view);
        btn.onclick = () => { state.view = btn.dataset.view; render(); };
      });
    }

    function bboxHtml(img) {
      return '';
    }

    function faceCard(title, img, tone = '') {
      if (!img) {
        return `<div class="face-card"><div class="face-label">${esc(title)}</div><div class="img-box">无数据</div></div>`;
      }
      const score = img.score === null || img.score === undefined ? '' : `<div class="face-score">相似度 ${fmt(img.score)}</div>`;
      return `
        <div class="face-card ${tone}">
          <div class="face-label">${esc(title)}</div>
          <div class="img-box"><img src="${img.src}" alt="${esc(img.name)}">${bboxHtml(img)}</div>
          <div class="face-name">${esc(img.identityDisplay || '-')}</div>
          <div class="face-file" title="${esc(img.path)}">${esc(img.name)}</div>
          ${score}
        </div>`;
    }

    function vectorDetails(label, img) {
      if (!img || !img.featurePreview) return '';
      return `<details class="tech"><summary>${esc(label)}：${img.featureDim} 维向量缩略</summary><pre>${esc(JSON.stringify(img.featurePreview))}\n\n完整向量未写入 HTML，避免全量看板卡顿。需要完整向量请查看缓存 JSON。</pre></details>`;
    }

    function groupByPerson(cases, filter) {
      const groups = new Map();
      const q = (filter || '').toLowerCase();
      cases.forEach(c => {
        const haystack = (c.identityDisplay + ' ' + c.queryName + ' ' + c.top1IdentityDisplay).toLowerCase();
        if (q && !haystack.includes(q)) return;
        if (!groups.has(c.identity)) {
          groups.set(c.identity, {
            identity: c.identity,
            identityDisplay: c.identityDisplay,
            cases: [],
            top1: 0,
            topk: 0,
            miss: 0,
            minScore: null,
            maxRisk: null
          });
        }
        const g = groups.get(c.identity);
        g.cases.push(c);
        if (c.correctRank === 1) g.top1 += 1;
        if (c.correctRank && c.correctRank <= DATA.topK) g.topk += 1;
        if (!c.correctRank || c.correctRank > DATA.topK) g.miss += 1;
        if (c.correctScore !== null && c.correctScore !== undefined) g.minScore = g.minScore === null ? c.correctScore : Math.min(g.minScore, c.correctScore);
        if (c.top1Score !== null && c.top1Score !== undefined) g.maxRisk = g.maxRisk === null ? c.top1Score : Math.max(g.maxRisk, c.top1Score);
      });
      return [...groups.values()].sort((a, b) => b.cases.length - a.cases.length || a.identityDisplay.localeCompare(b.identityDisplay));
    }

    function renderDetail(c) {
      if (!c) return `<div class="panel">没有可展示案例。</div>`;
      const good = c.correctRank === 1;
      const warn = c.correctRank && c.correctRank > 1 && c.correctRank <= DATA.topK;
      const badge = good ? 'good' : warn ? 'warn' : 'bad';
      const correctTitle = c.correct ? '正确身份底库' : '正确身份底库（未召回）';
      return `
        <div class="detail-head">
          <div>
            <h2>${esc(c.identityDisplay)} / ${esc(c.queryName)}</h2>
            <div class="case-meta">
              正确排名：${esc(c.correctRank ?? 'Top' + DATA.topK + '外')}　
              Top1：${esc(c.top1IdentityDisplay || '-')} ${fmt(c.top1Score)}　
              正确相似度：${fmt(c.correctScore)}　
              分差：${fmt(c.gap)}
            </div>
          </div>
          <span class="badge ${badge}">${esc(c.status)}</span>
        </div>
        <div class="compare-grid">
          ${faceCard('查询图', c.query)}
          ${faceCard('Top1 返回结果', c.top1, good ? 'good' : 'warn')}
          ${faceCard(correctTitle, c.correct, 'good')}
        </div>
        <div class="strip-panel">
          <h3>Top${DATA.topK} 候选缩略图</h3>
          <div class="top-strip">
            ${c.topMatches.map(m => `
              <div class="thumb ${m.isCorrect ? 'correct' : ''}">
                <div class="thumb-img"><img src="${m.image.src}" alt="${esc(m.image.name)}">${bboxHtml(m.image)}</div>
                <div class="thumb-rank">Top${m.rank}${m.isCorrect ? ' / 正确' : ''}</div>
                <div class="thumb-name">${esc(m.identityDisplay)}</div>
                <div class="thumb-score">${fmt(m.score)}</div>
              </div>`).join('')}
          </div>
        </div>
        ${vectorDetails('查询图', c.query)}
        ${vectorDetails('Top1 底库', c.top1)}
        ${c.correct && (!c.top1 || c.correct.identity !== c.top1.identity) ? vectorDetails('正确身份底库', c.correct) : ''}
      `;
    }

    function miniFace(label, img) {
      if (!img) {
        return `<div class="mini-face"><div class="mini-label">${esc(label)}</div><div class="mini-img">无</div></div>`;
      }
      const score = img.score === null || img.score === undefined ? '' : `<div class="mini-score">相似度 ${fmt(img.score)}</div>`;
      return `
        <div class="mini-face">
          <div class="mini-label">${esc(label)}</div>
          <div class="mini-img"><img src="${img.src}" alt="${esc(img.name)}"></div>
          <div class="mini-name" title="${esc(img.identityDisplay || img.name)}">${esc(img.identityDisplay || img.name)}</div>
          ${score}
        </div>`;
    }

    function resultBlock(c) {
      const good = c.correctRank === 1;
      const warn = c.correctRank && c.correctRank > 1 && c.correctRank <= DATA.topK;
      const tone = good ? 'good' : warn ? 'warn' : 'bad';
      return `
        <article class="mini-card ${tone}">
          <div class="mini-head">
            <div>
              <div class="mini-title">${esc(c.queryName)}</div>
              <div class="mini-meta">
                正确排名 ${esc(c.correctRank ?? 'Top' + DATA.topK + '外')}　
                Top1 ${esc(c.top1IdentityDisplay || '-')} ${fmt(c.top1Score)}　
                正确相似度 ${fmt(c.correctScore)}
              </div>
            </div>
            <span class="mini-badge ${tone}">${esc(c.status)}</span>
          </div>
          <div class="mini-compare">
            ${miniFace('查询图', c.query)}
            ${miniFace('Top1', c.top1)}
            ${miniFace('正确身份', c.correct)}
          </div>
          <div class="mini-actions">
            <details>
              <summary>查看 Top${DATA.topK} 候选</summary>
              <div class="mini-top10">
                ${c.topMatches.map(m => `
                  <div class="mini-thumb ${m.isCorrect ? 'correct' : ''}">
                    <img src="${m.image.src}" alt="${esc(m.image.name)}">
                    <div>Top${m.rank}${m.isCorrect ? ' 正确' : ''}</div>
                    <div>${esc(m.identityDisplay)}</div>
                    <div>${fmt(m.score)}</div>
                  </div>`).join('')}
              </div>
            </details>
            <details>
              <summary>查看向量</summary>
              ${vectorDetails('查询图', c.query)}
              ${vectorDetails('Top1 底库', c.top1)}
              ${c.correct && (!c.top1 || c.correct.identity !== c.top1.identity) ? vectorDetails('正确身份底库', c.correct) : ''}
            </details>
          </div>
        </article>`;
    }

    function sortCasesForPerson(cases) {
      return [...cases].sort((a, b) => {
        const ar = a.correctRank ?? 9999;
        const br = b.correctRank ?? 9999;
        if (ar !== br) return br - ar;
        return (b.top1Score ?? -1) - (a.top1Score ?? -1);
      });
    }

    function renderPersonResults(group, key, title) {
      if (!group) return `<div class="panel">没有可展示人物。</div>`;
      const cases = sortCasesForPerson(group.cases);
      return `
        <div class="detail-head">
          <div>
            <h2>${esc(group.identityDisplay)} · ${esc(title)}</h2>
            <div class="case-meta">
              当前分类 ${group.cases.length} 条　
              Top1 ${group.top1}　
              Top${DATA.topK} ${group.topk}　
              未命中 ${group.miss}　
              最低正确相似度 ${fmt(group.minScore)}
            </div>
          </div>
        </div>
        <div class="result-grid">
          ${cases.map(resultBlock).join('')}
        </div>`;
    }

    function renderCaseView(key, title, description) {
      const cases = DATA.views[key] || [];
      const filter = (state.filters[key] || '').toLowerCase();
      const groups = groupByPerson(cases, filter);
      if (!state.selectedPerson[key] || !groups.some(g => g.identity === state.selectedPerson[key])) {
        state.selectedPerson[key] = groups[0]?.identity || '';
      }
      const selectedGroup = groups.find(g => g.identity === state.selectedPerson[key]);
      return `
        <div class="workspace">
          <aside class="case-list">
            <div class="list-head">
              <h2>${esc(title)} <span class="nav-count">${DATA.counts[key]} 条 / ${groups.length} 人</span></h2>
              <div class="case-meta">${esc(description)}</div>
              <input placeholder="按人物、图片名、Top1身份搜索" value="${esc(state.filters[key] || '')}" data-filter="${key}">
            </div>
            <div class="list-items">
              ${groups.map(g => `
                <button class="case-item ${g.identity === state.selectedPerson[key] ? 'active' : ''}" data-person="${esc(g.identity)}" data-case-key="${key}">
                  <div class="case-title">${esc(g.identityDisplay)}</div>
                  <div class="case-meta">${g.cases.length} 条　Top1 ${g.top1}　Top${DATA.topK} ${g.topk}　未命中 ${g.miss}</div>
                  <div class="case-meta">最低正确相似度 ${fmt(g.minScore)}　最高Top1 ${fmt(g.maxRisk)}</div>
                </button>`).join('') || '<div class="panel">没有匹配结果。</div>'}
            </div>
          </aside>
          <section class="detail-pane">${renderPersonResults(selectedGroup, key, title)}</section>
        </div>`;
    }

    function renderOverview() {
      const m = DATA.metrics;
      return `
        <div class="overview-grid">
          <div class="metric-card"><div class="k">Top1准确率</div><div class="v">${pct(m.top1_accuracy_valid)}</div></div>
          <div class="metric-card"><div class="k">Top5准确率</div><div class="v">${pct(m.top5_accuracy_valid)}</div></div>
          <div class="metric-card"><div class="k">Top${DATA.topK}准确率</div><div class="v">${pct(m.top10_accuracy_valid)}</div></div>
          <div class="metric-card"><div class="k">有效查询 / 失败</div><div class="v">${m.query_valid} / ${m.query_failed}</div></div>
          <div class="metric-card"><div class="k">底库人数</div><div class="v">${DATA.enrollmentPeople}</div></div>
          <div class="metric-card"><div class="k">子集图片</div><div class="v">${DATA.subsetImages}</div></div>
          <div class="metric-card"><div class="k">正确身份中位相似度</div><div class="v">${fmt(m.median_correct_score)}</div></div>
          <div class="metric-card"><div class="k">Top1中位相似度</div><div class="v">${fmt(m.median_top1_score)}</div></div>
        </div>
        <div class="panel">
          <h2>测试口径</h2>
          <p>闭集 1:N 人脸识别：每人 1 张底库注册照，其余图片作为查询图。查询图一定属于这 ${DATA.enrollmentPeople} 个底库身份之一，本报告不评估库外陌生人拒识。</p>
          <p>接口：${esc(DATA.apiUrl)}；生成时间：${esc(DATA.generatedAt)}；图片已嵌入 HTML，可直接拷贝查看。</p>
        </div>
        <div class="panel">
          <h2>Rank 命中率</h2>
          <table><thead><tr><th>Rank</th><th class="num">命中数</th><th class="num">有效查询口径</th></tr></thead><tbody>
            ${Object.entries(m.rank_hits).map(([k, v]) => `<tr><td>Rank-${k}</td><td class="num">${v}</td><td class="num">${pct(v / m.query_valid)}</td></tr>`).join('')}
          </tbody></table>
        </div>`;
    }

    function renderPeople() {
      const rows = DATA.personStats || [];
      return `<div class="panel"><h2>按人统计</h2><table><thead><tr><th>人物</th><th class="num">图片数</th><th class="num">有效查询</th><th class="num">失败</th><th class="num">Top1</th><th class="num">Top${DATA.topK}</th><th class="num">中位正确相似度</th><th>最差样本</th><th class="num">最差排名</th></tr></thead><tbody>
        ${rows.map(r => `<tr><td>${esc(r.identityDisplay)}</td><td class="num">${r.images}</td><td class="num">${r.query_valid}</td><td class="num">${r.query_failed}</td><td class="num">${pct(r.top1_accuracy)}</td><td class="num">${pct(r.topk_accuracy)}</td><td class="num">${fmt(r.median_correct_score)}</td><td>${esc(r.worst_query || '')}</td><td class="num">${esc(r.worst_rank ?? '-')}</td></tr>`).join('')}
      </tbody></table></div>`;
    }

    function renderFailures() {
      const rows = DATA.failures || [];
      return `<div class="panel"><h2>失败图片</h2><table><thead><tr><th>人物</th><th>路径</th><th>原因</th></tr></thead><tbody>
        ${rows.length ? rows.map(f => `<tr><td>${esc((f.identity || '').replaceAll('_', ' '))}</td><td>${esc(f.path || '')}</td><td>${esc(f.error || '')}</td></tr>`).join('') : '<tr><td colspan="3">无</td></tr>'}
      </tbody></table></div>`;
    }

    function renderNotes() {
      return `<div class="panel"><h2>测试说明</h2>
        <p>本看板按业务评审方式组织：先看正确 Top 命中，再看 Top10 可召回，最后看高风险误识别和 Top10 未命中。</p>
        <p>相似度为归一化向量 cosine。绿色表示正确身份，橙色表示 Top1 错误，红色表示 Top${DATA.topK} 未命中。</p>
        <p>完整向量默认隐藏，只在案例详情底部展开查看，避免干扰肉眼比对。</p>
      </div>`;
    }

    function render() {
      nav();
      let html = '';
      if (state.view === 'overview') html = renderOverview();
      if (state.view === 'success') html = renderCaseView('success', 'Top1正确命中', '正确身份排名第 1 的全部样例。按人物聚合，点击人物查看该人物所有查询结果。');
      if (state.view === 'top5') html = renderCaseView('top5', 'Top5正确命中', '正确身份排在 Top5 内的全部样例，包含 Top1 正确命中。');
      if (state.view === 'top10') html = renderCaseView('top10', 'Top10正确命中', '正确身份排在 Top10 内的全部样例，包含 Top1 和 Top5 正确命中。');
      if (state.view === 'risk') html = renderCaseView('risk', '高风险误识别', 'Top1 错误且相似度较高的样例优先展示。');
      if (state.view === 'miss') html = renderCaseView('miss', 'Top10未命中', '正确身份没有进入 Top10 的失败样例。');
      if (state.view === 'people') html = renderPeople();
      if (state.view === 'failures') html = renderFailures();
      if (state.view === 'notes') html = renderNotes();
      $('#content').innerHTML = `<section class="view active">${html}</section>`;
      $$('[data-person]').forEach(btn => btn.onclick = () => { state.selectedPerson[btn.dataset.caseKey] = btn.dataset.person; render(); });
      $$('[data-filter]').forEach(input => input.oninput = () => { state.filters[input.dataset.filter] = input.value; state.selectedPerson[input.dataset.filter] = ''; render(); });
    }

    document.addEventListener('keydown', (e) => {
      return;
    });
    render();
    """


def make_dashboard_html(args, distribution, subset, enrollments, enrollment_failures, rows, query_failures, metrics, person_stats, generated_at):
    data = build_dashboard_data(args, distribution, subset, enrollments, enrollment_failures, rows, query_failures, metrics, person_stats, generated_at)
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    m = metrics
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(args.title)}</title>
  <style>{dashboard_style()}</style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div>
        <h1>{esc(args.title)}</h1>
        <div class="sub">闭集 1:N 识别 · {esc(generated_at)} · 图片已嵌入 HTML</div>
      </div>
      <div class="top-metrics">
        <div class="top-pill"><div class="k">Top1</div><div class="v">{pct(m['top1_hits'], m['query_valid'])}</div></div>
        <div class="top-pill"><div class="k">Top5</div><div class="v">{pct(m['top5_hits'], m['query_valid'])}</div></div>
        <div class="top-pill"><div class="k">Top{args.top_k}</div><div class="v">{pct(m['top10_hits'], m['query_valid'])}</div></div>
        <div class="top-pill"><div class="k">有效/失败</div><div class="v">{m['query_valid']}/{m['query_failed']}</div></div>
      </div>
    </header>
    <nav class="sidebar">
      <button class="nav-btn" data-view="overview"><span>总览</span><span class="nav-count">指标</span></button>
      <button class="nav-btn" data-view="success"><span>Top1正确命中</span><span class="nav-count">{data['counts']['success']}</span></button>
      <button class="nav-btn" data-view="top5"><span>Top5正确命中</span><span class="nav-count">{data['counts']['top5']}</span></button>
      <button class="nav-btn" data-view="top10"><span>Top10正确命中</span><span class="nav-count">{data['counts']['top10']}</span></button>
      <button class="nav-btn" data-view="risk"><span>高风险误识别</span><span class="nav-count">{data['counts']['risk']}</span></button>
      <button class="nav-btn" data-view="miss"><span>Top10未命中</span><span class="nav-count">{data['counts']['miss']}</span></button>
      <button class="nav-btn" data-view="people"><span>按人统计</span><span class="nav-count">{len(person_stats)}</span></button>
      <button class="nav-btn" data-view="failures"><span>失败图片</span><span class="nav-count">{len(enrollment_failures) + len(query_failures)}</span></button>
      <button class="nav-btn" data-view="notes"><span>测试说明</span><span class="nav-count">附录</span></button>
    </nav>
    <main id="content" class="content"></main>
  </div>
  <script id="report-data" type="application/json">{data_json}</script>
  <script>{dashboard_script()}</script>
</body>
</html>
"""


def make_html(args, distribution, subset, enrollments, enrollment_failures, rows, query_failures, metrics, person_stats, generated_at):
    return make_dashboard_html(args, distribution, subset, enrollments, enrollment_failures, rows, query_failures, metrics, person_stats, generated_at)

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
