import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from face_ais_bench import FaceModelAISBench


def _load_image(path: Path) -> np.ndarray:
    data = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to decode image: {path}")
    return img


def _set_default_env():
    os.environ.setdefault("DET_INPUT_SHAPE", "640,640")
    os.environ.setdefault("KPS_INPUT_SHAPE", "192,192")
    os.environ.setdefault("REC_INPUT_SHAPE", "112,112")
    os.environ.setdefault("ATTR_INPUT_SHAPE", "96,96")


def main():
    parser = argparse.ArgumentParser(description="AISBench vector sanity check")
    parser.add_argument("--image", default="test.png", help="image path (default: test.png)")
    parser.add_argument("--models-dir", default=None, help="models root containing per-SoC directories")
    parser.add_argument("--save-emb", default=None, help="optional npy output path")
    args = parser.parse_args()

    _set_default_env()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    img = _load_image(image_path)
    model = FaceModelAISBench(models_dir=args.models_dir)
    results = model.process_image(img, extract_features=True, predict_attributes=False)

    print(f"faces: {len(results)}")
    if not results:
        return

    embeddings = []
    for idx, r in enumerate(results):
        feat = np.array(r["feature"], dtype=np.float32)
        embeddings.append(feat)
        norm = float(np.linalg.norm(feat))
        head = np.array2string(feat[:10], precision=4, separator=",")
        print(f"face[{idx}] feature_len={feat.shape[0]} norm={norm:.6f} head={head}")

    if args.save_emb:
        out_path = Path(args.save_emb)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.stack(embeddings, axis=0))
        print(f"saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
