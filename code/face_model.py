"""
Face Recognition Model Implementation using ONNXRuntime.

This module loads the buffalo_l ONNX models for detection, keypoints,
recognition, and attributes, then runs a full face analysis pipeline.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from face_ais_bench import FaceModelAISBench

logger = logging.getLogger(__name__)


class FaceModelONNX:
    """Face recognition pipeline powered by ONNXRuntime."""

    def __init__(self, models_dir: str = None, providers: List[str] = None):
        start_time = time.time()
        self.gender_labels = ["Male", "Female"]
        self.race_labels = ["Asian", "White", "Black", "Indian", "Other"]
        self.face_attributes = []
        self.face_index = 0
        self.providers = self._resolve_providers(providers)

        base_dir = Path(models_dir) if models_dir else self._default_models_dir()
        if not base_dir.exists():
            raise FileNotFoundError(f"models directory not found: {base_dir}")

        logger.info("ONNX models dir: %s", base_dir)
        self.det_sess = self._load_session(base_dir / "det_10g.onnx", required=True)
        self.kps_sess = self._load_session(base_dir / "2d106det.onnx")
        self.rec_sess = self._load_session(base_dir / "w600k_r50.onnx")
        self.attr_sess = self._load_session(base_dir / "genderage.onnx")

        logger.info("ONNX providers: %s", self.providers)
        logger.info("Initialization completed in %.2f seconds", time.time() - start_time)

    def _default_models_dir(self) -> Path:
        env_dir = os.environ.get("FACE_ONNX_DIR")
        if env_dir:
            return Path(env_dir)
        return Path(__file__).resolve().parent.parent / "models" / "buffalo_l"

    def _resolve_providers(self, providers: List[str]):
        if providers:
            return providers
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_session(self, model_path: Path, required: bool = False):
        if not model_path.exists():
            if required:
                raise FileNotFoundError(f"model not found: {model_path}")
            logger.warning("Model not found: %s", model_path)
            return None
        sess = ort.InferenceSession(str(model_path), providers=self.providers)
        logger.info("Loaded model: %s", model_path.name)
        return sess

    def _infer(self, sess: ort.InferenceSession, inp: np.ndarray, tag: str = None) -> List[np.ndarray]:
        input_name = sess.get_inputs()[0].name
        start = time.perf_counter()
        outs = sess.run(None, {input_name: inp})
        cost_ms = (time.perf_counter() - start) * 1000.0
        if tag:
            logger.info("%s infer: %.2f ms", tag, cost_ms)
        return outs

    def _safe_int(self, value):
        try:
            return int(value)
        except Exception:
            return None

    def _get_input_hw(self, sess, env_name: str, default: Tuple[int, int] = None) -> Tuple[int, int]:
        env_shape = os.environ.get(env_name)
        if env_shape:
            h, w = [int(x) for x in env_shape.split(",")]
            return h, w
        shape = sess.get_inputs()[0].shape
        h = self._safe_int(shape[-2])
        w = self._safe_int(shape[-1])
        if h is None or w is None:
            if default is None:
                raise ValueError(f"{env_name} not set and model input shape is dynamic")
            return default
        return h, w

    def _align_112(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2, _ = bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            face = img
        return cv2.resize(face, (112, 112))

    def _align_face(self, img: np.ndarray, pts5: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        if pts5 is None or pts5.shape != (5, 2):
            return None
        dst = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if out_size != (112, 112):
            scale_x = out_size[0] / 112.0
            scale_y = out_size[1] / 112.0
            dst = dst.copy()
            dst[:, 0] *= scale_x
            dst[:, 1] *= scale_y
        m, _ = cv2.estimateAffinePartial2D(pts5.astype(np.float32), dst, method=cv2.LMEDS)
        if m is None:
            return None
        return cv2.warpAffine(img, m, out_size, borderValue=0.0)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5, topk: int = 100) -> List[int]:
        idxs = scores.argsort()[::-1]
        keep = []
        while idxs.size > 0 and len(keep) < topk:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
            iou = inter / (area_i + area_j - inter + 1e-6)
            idxs = idxs[1:][iou <= iou_thr]
        return keep

    def _preprocess_det(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        h, w = self._get_input_hw(self.det_sess, "DET_INPUT_SHAPE", default=(640, 640))
        ih, iw = img.shape[0], img.shape[1]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (nw, nh))
        pad = np.zeros((h, w, 3), dtype=np.float32)
        pad[:nh, :nw] = x
        x = pad
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return x, (scale, scale), (h, w)

    def _decode_scrfd(self, scores: np.ndarray, bboxes: np.ndarray, stride: int, det_shape: Tuple[int, int]):
        det_h, det_w = det_shape
        feat_h = int(det_h / stride)
        feat_w = int(det_w / stride)
        anchor_num = int(bboxes.shape[0] / (feat_h * feat_w))
        if anchor_num < 1:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        xs = np.arange(feat_w)
        ys = np.arange(feat_h)
        xs, ys = np.meshgrid(xs, ys)
        centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
        centers = (centers + 0.5) * stride
        if anchor_num > 1:
            centers = np.repeat(centers, anchor_num, axis=0)
        scores = scores.reshape(-1).astype(np.float32)
        if scores.size != centers.shape[0]:
            size = min(scores.size, centers.shape[0])
            scores = scores[:size]
            bboxes = bboxes[:size]
            centers = centers[:size]
        bboxes = bboxes.astype(np.float32) * stride
        x1 = centers[:, 0] - bboxes[:, 0]
        y1 = centers[:, 1] - bboxes[:, 1]
        x2 = centers[:, 0] + bboxes[:, 2]
        y2 = centers[:, 1] + bboxes[:, 3]
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        if scores.size and (scores.max() > 1.0 or scores.min() < 0.0):
            scores = 1.0 / (1.0 + np.exp(-scores))
        return boxes, scores

    def _postprocess_det(self, outs: List[np.ndarray], det_shape: Tuple[int, int], scale: Tuple[float, float]) -> List[List[float]]:
        scores_list = []
        bbox_list = []
        for o in outs:
            a = o
            if a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
            if a.ndim == 2 and a.shape[-1] == 1:
                scores_list.append(a)
            elif a.ndim == 2 and a.shape[-1] == 4:
                bbox_list.append(a)
        if not scores_list or not bbox_list:
            return []
        scores_list = sorted(scores_list, key=lambda x: x.shape[0], reverse=True)
        bbox_list = sorted(bbox_list, key=lambda x: x.shape[0], reverse=True)
        strides = [8, 16, 32]
        all_boxes = []
        all_scores = []
        for idx, (scores, bboxes) in enumerate(zip(scores_list, bbox_list)):
            stride = strides[idx] if idx < len(strides) else strides[-1]
            boxes, scr = self._decode_scrfd(scores, bboxes, stride, det_shape)
            if boxes.size == 0:
                continue
            score_thr = float(os.environ.get("DET_SCORE_THRESH", "0.5"))
            mask = scr >= score_thr
            if not np.any(mask):
                continue
            all_boxes.append(boxes[mask])
            all_scores.append(scr[mask])
        if not all_boxes:
            return []
        b = np.concatenate(all_boxes, axis=0)
        s = np.concatenate(all_scores, axis=0)
        b[:, [0, 2]] = b[:, [0, 2]] * (1.0 / scale[0])
        b[:, [1, 3]] = b[:, [1, 3]] * (1.0 / scale[1])
        keep = self._nms(b, s, float(os.environ.get("DET_NMS_IOU", "0.5")), int(os.environ.get("DET_TOPK", "100")))
        res = []
        for i in keep:
            x1, y1, x2, y2 = b[i].tolist()
            res.append([x1, y1, x2, y2, float(s[i])])
        return res

    def _preprocess_kps(self, face: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = self._get_input_hw(self.kps_sess, "KPS_INPUT_SHAPE", default=(192, 192))
        x = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (w, h))
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return x, (h, w)

    def _postprocess_kps(self, outs: List[np.ndarray], w: int, h: int) -> np.ndarray:
        for o in outs:
            a = o.squeeze()
            if a.ndim == 2 and a.shape[0] == 106 and a.shape[1] == 2:
                pts = a.copy()
                pts[:, 0] = pts[:, 0] * w
                pts[:, 1] = pts[:, 1] * h
                return pts
            if a.ndim == 1 and a.shape[0] == 212:
                pts = a.reshape(106, 2).copy()
                pts[:, 0] = pts[:, 0] * w
                pts[:, 1] = pts[:, 1] * h
                return pts
        return np.zeros((106, 2), dtype=np.float32)

    def get_input(self, img: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        if self.det_sess is None:
            raise RuntimeError("det_10g.onnx not loaded")
        inp, scale, det_shape = self._preprocess_det(img)
        outs = self._infer(self.det_sess, inp, tag="det")
        dets = self._postprocess_det(outs, det_shape, scale)
        if not dets:
            self.face_attributes = []
            self.face_index = 0
            return 0, np.array([]), np.array([]), np.array([])
        bboxes = dets
        pts5 = []
        aligned = []
        self.face_attributes = []
        self.face_index = 0
        for bb in bboxes:
            x1, y1, x2, y2, _ = bb
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img.shape[1], int(x2))
            y2 = min(img.shape[0], int(y2))
            crop = img[y1:y2, x1:x2]
            aligned_face = None
            if self.kps_sess is not None and crop.size != 0:
                kinp, kshape = self._preprocess_kps(crop)
                kouts = self._infer(self.kps_sess, kinp, tag="kps")
                pts106 = self._postprocess_kps(kouts, kshape[1], kshape[0])
                cw = max(1, x2 - x1)
                ch = max(1, y2 - y1)
                pts106[:, 0] = pts106[:, 0] * (cw / kshape[1]) + x1
                pts106[:, 1] = pts106[:, 1] * (ch / kshape[0]) + y1
                sel_align = np.array(
                    [pts106[8], pts106[36], pts106[30], pts106[54], pts106[48]],
                    dtype=np.float32,
                )
                aligned_face = self._align_face(img, sel_align)
                sel_out = [pts106[30], pts106[8], pts106[36], pts106[54], pts106[48]]
                flat = [c for p in sel_out for c in p]
                pts5.append(flat)
            else:
                pts5.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            if aligned_face is None:
                aligned_face = self._align_112(img, np.array(bb))
            aligned.append(aligned_face)
            self.face_attributes.append(("Unknown", 0, "Unknown"))
        return len(bboxes), np.array(bboxes), np.array(pts5), np.array(aligned)

    def _preprocess_rec(self, face: np.ndarray) -> np.ndarray:
        h, w = self._get_input_hw(self.rec_sess, "REC_INPUT_SHAPE", default=(112, 112))
        x = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (w, h))
        x = x.astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return x

    def _preprocess_attr(self, face: np.ndarray) -> np.ndarray:
        h, w = self._get_input_hw(self.attr_sess, "ATTR_INPUT_SHAPE", default=(96, 96))
        x = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (w, h))
        x = x.astype(np.float32)
        x = x / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return x

    def get_feature(self, aligned_face: np.ndarray) -> np.ndarray:
        if self.rec_sess is None:
            emb = aligned_face.astype(np.float32).flatten()
            n = np.linalg.norm(emb) + 1e-6
            return emb / n
        inp = self._preprocess_rec(aligned_face)
        outs = self._infer(self.rec_sess, inp, tag="rec")
        feat = outs[0]
        if feat.ndim > 2:
            feat = feat.reshape(feat.shape[0], -1)
        feat = feat.squeeze(0)
        n = np.linalg.norm(feat) + 1e-6
        return (feat / n).astype(np.float32)

    def get_gender_age_race(self, aligned_face: np.ndarray):
        if self.attr_sess is None or not self.face_attributes or self.face_index >= len(self.face_attributes):
            return "Unknown", 0, "Unknown"
        inp = self._preprocess_attr(aligned_face)
        outs = self._infer(self.attr_sess, inp, tag="attr")
        gender = "Unknown"
        age = 0
        race = "Unknown"
        for a in outs:
            v = a.squeeze()
            if v.ndim == 1 and v.shape[0] == 3:
                gender = self.gender_labels[int(np.argmax(v[:2]))]
                age_val = float(v[2])
                age = int(age_val * 100) if age_val <= 1.0 else int(age_val)
            elif v.ndim == 1 and v.shape[0] == 2:
                gender = self.gender_labels[int(np.argmax(v))]
            elif v.ndim == 1 and v.shape[0] in (4, 5):
                race = self.race_labels[int(np.argmax(v))]
            elif v.ndim == 0 or (v.ndim == 1 and v.shape[0] == 1):
                age = int(float(v))
        if gender == "Unknown" or race == "Unknown":
            gender0, age0, race0 = self.face_attributes[self.face_index]
            if gender == "Unknown":
                gender = gender0
            if age == 0:
                age = age0
            if race == "Unknown":
                race = race0
        self.face_index += 1
        return gender, age, race

    def process_image(self, img: np.ndarray, extract_features: bool = True, predict_attributes: bool = True):
        results = []
        num, bboxes, landmarks, aligned = self.get_input(img)
        for i in range(num):
            r = {"bbox": bboxes[i].tolist(), "landmark": landmarks[i].tolist()}
            if extract_features:
                r["feature"] = self.get_feature(aligned[i]).tolist()
            if predict_attributes:
                g, a, rc = self.get_gender_age_race(aligned[i])
                r["gender"] = g
                r["age"] = a
                r["race"] = rc
            results.append(r)
        return results


__all__ = ["FaceModelONNX", "FaceModelAISBench"]

