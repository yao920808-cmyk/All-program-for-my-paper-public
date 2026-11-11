from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, roc_auc_score
from skimage.color import rgb2hed
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 42
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FLIP_STATE_VALUES = (0, 1, 2, 3)
ALL_ORIENTATION_STATE_VALUES = tuple(range(8))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def print_status(message: str) -> None:
    print(message, flush=True)


def format_elapsed(seconds: float) -> str:
    total_seconds = max(float(seconds), 0.0)
    if total_seconds < 60.0:
        return f"{total_seconds:.1f}s"
    minutes, rem_seconds = divmod(total_seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{rem_seconds:04.1f}s"
    hours, rem_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(rem_minutes):02d}m{rem_seconds:04.1f}s"


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot read image because the file does not exist: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Cannot read image bytes from: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"Failed to encode image for saving: {path}")
    encoded.tofile(str(path))


def list_image_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Training folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Training path is not a folder: {folder}")

    img_paths = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return img_paths


def get_default_image_dir() -> Path:
    candidate = Path.cwd() / "image_file"
    return candidate if candidate.is_dir() else Path.cwd()


def default_slice_root() -> Path:
    candidate = Path.cwd() / "cutting image"
    return candidate if candidate.is_dir() else Path.cwd()


def orientation_state(rotation_degrees: int, horizontal_flip: bool) -> int:
    normalized_rotation = rotation_degrees % 360
    if normalized_rotation not in (0, 90, 180, 270):
        raise ValueError(f"Unsupported rotation: {rotation_degrees}")
    return normalized_rotation // 90 + (4 if horizontal_flip else 0)


def state_to_rotation_flip(state: int) -> Tuple[int, bool]:
    if state not in ALL_ORIENTATION_STATE_VALUES:
        raise ValueError(f"Unknown state: {state}")
    return (state % 4) * 90, state >= 4


def inverse_state(state: int) -> int:
    rotation_degrees, horizontal_flip = state_to_rotation_flip(state)
    if horizontal_flip:
        return state
    return orientation_state((-rotation_degrees) % 360, False)


def parse_bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def read_slice_metadata(folder: str | Path) -> List[Dict[str, str]] | None:
    metadata_path = Path(folder) / "metadata.csv"
    if not metadata_path.is_file():
        return None
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def gridify_tiles(tiles: Sequence[np.ndarray], rows: int, cols: int) -> List[List[np.ndarray]]:
    if len(tiles) != rows * cols:
        raise ValueError(f"Expected {rows * cols} tiles, got {len(tiles)}")
    return [list(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]


def build_visual_grid(tiles_grid: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    flat_tiles = [tile for row in tiles_grid for tile in row]
    if not flat_tiles:
        raise ValueError("Cannot build a grid from zero tiles.")
    target_h = min(tile.shape[0] for tile in flat_tiles)
    target_w = min(tile.shape[1] for tile in flat_tiles)
    resized_rows = []
    for row in tiles_grid:
        resized_row = [
            cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for tile in row
        ]
        resized_rows.append(np.concatenate(resized_row, axis=1))
    return np.concatenate(resized_rows, axis=0)


def find_slice_folder_for_image(
    image_path: str | Path,
    rows: int,
    cols: int,
    slice_root: str | Path | None = None,
) -> Path | None:
    image_path = Path(image_path)
    search_root = Path(slice_root) if slice_root is not None else default_slice_root()
    grid_tokens = (f"{rows}x{cols}", f"{cols}x{rows}")
    candidates = [
        search_root / f"{image_path.stem}_{grid_token}"
        for grid_token in grid_tokens
    ]
    candidates.extend(
        [
            search_root / f"{grid_token}_randomized" / image_path.stem
            for grid_token in grid_tokens
        ]
    )
    candidates.extend(
        [
            search_root / f"{grid_token}_randomized" / f"{image_path.stem}_{grid_token}"
            for grid_token in grid_tokens
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    for grid_token in grid_tokens:
        randomized_root = search_root / f"{grid_token}_randomized"
        if not randomized_root.is_dir():
            continue
        for pattern in (f"{image_path.stem}_*", f"{image_path.stem}"):
            matches = sorted(path for path in randomized_root.glob(pattern) if path.is_dir())
            if matches:
                return matches[0]
    return None


def load_tiles_from_folder(
    folder: str | Path,
    rows: int | None = None,
    cols: int | None = None,
) -> List[np.ndarray]:
    folder = Path(folder)
    metadata_rows = read_slice_metadata(folder)
    if metadata_rows is not None:
        if rows is None or cols is None:
            raise ValueError(f"rows/cols are required to load randomized slice metadata from {folder}")
        ordered_tiles, _ = load_randomized_tiles_and_pieces(folder, rows, cols)
        return ordered_tiles
    tile_paths = list_image_paths(folder)
    return [read_image(path) for path in tile_paths]


def ensure_divisible_resize(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = (h // rows) * rows
    new_w = (w // cols) * cols
    if new_h == 0 or new_w == 0:
        raise ValueError("Image too small for requested grid size.")
    if new_h != h or new_w != w:
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def split_into_tiles(img: np.ndarray, rows: int, cols: int) -> List[np.ndarray]:
    img = ensure_divisible_resize(img, rows, cols)
    h, w = img.shape[:2]
    th, tw = h // rows, w // cols
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tile = img[r * th:(r + 1) * th, c * tw:(c + 1) * tw].copy()
            tiles.append(tile)
    return tiles


def merge_tiles(tiles_grid: List[List[np.ndarray]]) -> np.ndarray:
    rows = [np.concatenate(row, axis=1) for row in tiles_grid]
    return np.concatenate(rows, axis=0)


def apply_state(tile: np.ndarray, state: int) -> np.ndarray:
    rotation_degrees, horizontal_flip = state_to_rotation_flip(state)
    if rotation_degrees == 0:
        transformed = tile
    elif rotation_degrees == 90:
        transformed = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_degrees == 180:
        transformed = cv2.rotate(tile, cv2.ROTATE_180)
    elif rotation_degrees == 270:
        transformed = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Unsupported rotation: {rotation_degrees}")
    if horizontal_flip:
        transformed = cv2.flip(transformed, 1)
    return transformed


def get_edge_strip(tile: np.ndarray, side: str, strip: int = 16) -> np.ndarray:
    h, w = tile.shape[:2]
    strip = min(strip, h, w)
    if side == "left":
        return tile[:, :strip, :]
    if side == "right":
        return tile[:, w - strip:, :]
    if side == "top":
        return tile[:strip, :, :]
    if side == "bottom":
        return tile[h - strip:, :, :]
    raise ValueError(f"Unknown side: {side}")


def relation_to_id(relation: str) -> int:
    if relation == "right":
        return 0
    if relation == "down":
        return 1
    raise ValueError(f"Unsupported relation: {relation}")


def pair_edge_strips(
    tile_a: np.ndarray,
    tile_b: np.ndarray,
    relation: str,
    strip_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if relation == "right":
        return (
            get_edge_strip(tile_a, "right", strip_size),
            get_edge_strip(tile_b, "left", strip_size),
            0,
        )
    if relation == "down":
        return (
            get_edge_strip(tile_a, "bottom", strip_size),
            get_edge_strip(tile_b, "top", strip_size),
            1,
        )
    raise ValueError(f"Unsupported relation: {relation}")


def resize_strip(strip: np.ndarray, size: Tuple[int, int] = (64, 16)) -> np.ndarray:
    return cv2.resize(strip, size, interpolation=cv2.INTER_LINEAR)


def normalize_image(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


@dataclass(frozen=True)
class StainNormalizer:
    method: str
    target_mean: np.ndarray
    target_std: np.ndarray
    reference_image_count: int
    reference_pixel_count: int

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    def normalize(self, img: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return img.copy()

        rgb = normalize_image(img)
        mask = build_tissue_mask(rgb)
        if int(mask.sum()) < 16:
            return img.copy()

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        source_pixels = lab[mask]
        source_mean = source_pixels.mean(axis=0)
        source_std = source_pixels.std(axis=0)

        normalized = lab.copy()
        normalized_pixels = ((source_pixels - source_mean) / (source_std + 1e-6)) * self.target_std + self.target_mean
        normalized[mask] = normalized_pixels
        normalized[:, :, 0] = np.clip(normalized[:, :, 0], 0.0, 100.0)
        normalized[:, :, 1] = np.clip(normalized[:, :, 1], -127.0, 127.0)
        normalized[:, :, 2] = np.clip(normalized[:, :, 2], -127.0, 127.0)
        normalized_rgb = cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)
        return np.clip(np.round(normalized_rgb * 255.0), 0.0, 255.0).astype(np.uint8)


def build_tissue_mask(img: np.ndarray) -> np.ndarray:
    rgb = img.astype(np.float32)
    if float(np.max(rgb)) > 1.0:
        rgb = normalize_image(rgb)
    brightness = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mask = (brightness < 0.95) & (chroma > 0.03)
    if int(mask.sum()) < 32:
        mask = brightness < 0.98
    if int(mask.sum()) < 32:
        mask = np.ones(mask.shape, dtype=bool)
    return mask


def sample_reference_image_paths(
    image_paths: Sequence[str | Path],
    sample_image_limit: int,
) -> List[Path]:
    resolved_paths = [Path(p).resolve() for p in image_paths]
    if sample_image_limit <= 0 or len(resolved_paths) <= sample_image_limit:
        return resolved_paths
    sampled_indices = np.linspace(0, len(resolved_paths) - 1, num=sample_image_limit, dtype=int)
    return [resolved_paths[idx] for idx in sorted(set(sampled_indices.tolist()))]


def build_stain_normalizer(
    image_paths: Sequence[str | Path],
    method: str = "none",
    sample_image_limit: int = 16,
    sample_pixels_per_image: int = 8192,
) -> StainNormalizer | None:
    method = str(method).strip().lower()
    if method == "none":
        return None
    if method != "reinhard":
        raise ValueError(f"Unsupported stain normalization method: {method}")

    sampled_paths = sample_reference_image_paths(image_paths, sample_image_limit)
    total_sum = np.zeros((3,), dtype=np.float64)
    total_sumsq = np.zeros((3,), dtype=np.float64)
    total_count = 0

    for image_path in sampled_paths:
        img = read_image(image_path)
        rgb = normalize_image(img)
        mask = build_tissue_mask(rgb)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        tissue_pixels = lab[mask].reshape(-1, 3)
        if tissue_pixels.size == 0:
            continue
        if sample_pixels_per_image > 0 and tissue_pixels.shape[0] > sample_pixels_per_image:
            rng = np.random.default_rng(stable_path_seed(image_path))
            select_idx = rng.choice(tissue_pixels.shape[0], size=sample_pixels_per_image, replace=False)
            tissue_pixels = tissue_pixels[select_idx]
        total_sum += tissue_pixels.sum(axis=0)
        total_sumsq += np.square(tissue_pixels).sum(axis=0)
        total_count += tissue_pixels.shape[0]

    if total_count == 0:
        raise ValueError("Stain normalization could not find enough tissue pixels to fit a reference profile.")

    target_mean = total_sum / total_count
    target_var = np.clip((total_sumsq / total_count) - np.square(target_mean), 1e-6, None)
    target_std = np.sqrt(target_var)
    return StainNormalizer(
        method=method,
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
        reference_image_count=len(sampled_paths),
        reference_pixel_count=int(total_count),
    )


def maybe_normalize_stain(img: np.ndarray, stain_normalizer: StainNormalizer | None) -> np.ndarray:
    if stain_normalizer is None or not stain_normalizer.enabled:
        return img
    return stain_normalizer.normalize(img)


def amp_autocast(device: str, enabled: bool, dtype: torch.dtype = torch.float16):
    if not enabled:
        return nullcontext()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=True, dtype=dtype)
    if device_type == "cuda" and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


def make_grad_scaler(device: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            device_type = "cuda" if str(device).startswith("cuda") else "cpu"
            return torch.amp.GradScaler(device_type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    if hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return None


def stain_deconvolution(strip_img: np.ndarray) -> np.ndarray:
    rgb = np.clip(normalize_image(strip_img), 1e-6, 1.0)
    hed = rgb2hed(rgb).astype(np.float32)
    return hed[:, :, :2]


def normalize_stain_channels(hed: np.ndarray) -> np.ndarray:
    return np.clip((hed + 1.5) / 3.0, 0.0, 1.0).astype(np.float32)


def strip_to_tensor(
    strip: np.ndarray,
    input_mode: str = "rgb_hed",
    stain_normalizer: StainNormalizer | None = None,
) -> np.ndarray:
    resized = resize_strip(strip, (64, 16))
    resized = maybe_normalize_stain(resized, stain_normalizer)
    rgb = normalize_image(resized)
    if input_mode == "rgb":
        return rgb.transpose(2, 0, 1)
    if input_mode == "rgb_hed":
        hed = normalize_stain_channels(stain_deconvolution(resized))
        return np.concatenate([rgb, hed], axis=2).transpose(2, 0, 1)
    raise ValueError(f"Unsupported CNN input_mode: {input_mode}")

FEATURE_MODES = ("color", "texture", "stain", "fusion")


def _normalized_hist(channel: np.ndarray, bins: int, value_range: Tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(channel.reshape(-1), bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-8)


def color_descriptor(
    strip_img: np.ndarray,
    stain_normalizer: StainNormalizer | None = None,
) -> np.ndarray:
    strip = maybe_normalize_stain(strip_img.copy(), stain_normalizer)
    lab = cv2.cvtColor(strip, cv2.COLOR_RGB2LAB)
    rgb = normalize_image(strip)
    feats = []

    for ch in range(3):
        hist = cv2.calcHist([lab], [ch], None, [8], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        feats.append(hist.astype(np.float32))

    for ch in range(3):
        channel = rgb[:, :, ch]
        feats.append(
            np.array(
                [
                    float(channel.mean()),
                    float(channel.std()),
                    float(np.percentile(channel, 25)),
                    float(np.percentile(channel, 50)),
                    float(np.percentile(channel, 75)),
                ],
                dtype=np.float32,
            )
        )
    return np.concatenate(feats).astype(np.float32)


def stain_descriptor(
    strip_img: np.ndarray,
    stain_normalizer: StainNormalizer | None = None,
) -> np.ndarray:
    normalized_strip = maybe_normalize_stain(strip_img, stain_normalizer)
    hed = stain_deconvolution(normalized_strip)
    hed_norm = normalize_stain_channels(hed)
    feats = []
    for ch in range(hed.shape[2]):
        raw_channel = hed[:, :, ch]
        norm_channel = hed_norm[:, :, ch]
        feats.append(_normalized_hist(norm_channel, bins=8, value_range=(0.0, 1.0)))
        feats.append(
            np.array(
                [
                    float(raw_channel.mean()),
                    float(raw_channel.std()),
                    float(np.percentile(raw_channel, 25)),
                    float(np.percentile(raw_channel, 50)),
                    float(np.percentile(raw_channel, 75)),
                ],
                dtype=np.float32,
            )
        )
    flat_h = hed[:, :, 0].reshape(-1)
    flat_e = hed[:, :, 1].reshape(-1)
    if flat_h.size > 1 and np.std(flat_h) > 1e-8 and np.std(flat_e) > 1e-8:
        corr = float(np.corrcoef(flat_h, flat_e)[0, 1])
    else:
        corr = 0.0
    feats.append(np.array([corr], dtype=np.float32))
    return np.nan_to_num(np.concatenate(feats).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def texture_descriptor(
    strip_img: np.ndarray,
    stain_normalizer: StainNormalizer | None = None,
) -> np.ndarray:
    normalized_strip = maybe_normalize_stain(strip_img, stain_normalizer)
    gray = cv2.cvtColor(normalized_strip, cv2.COLOR_RGB2GRAY).astype(np.float32)
    feats = [np.array([gray.mean(), gray.std()], dtype=np.float32)]

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    feats.append(np.array([mag.mean(), mag.std()], dtype=np.float32))

    hist_ang, _ = np.histogram(ang, bins=8, range=(0, 360), weights=mag)
    hist_ang = hist_ang.astype(np.float32)
    hist_ang = hist_ang / (hist_ang.sum() + 1e-8)
    feats.append(hist_ang)

    gray_small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.uint8)
    lbp = local_binary_pattern(gray_small, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(11), range=(0, 10))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)
    feats.append(lbp_hist)

    quantized = np.clip(gray_small // 8, 0, 31).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0.0, np.pi / 2],
        levels=32,
        symmetric=True,
        normed=True,
    )
    glcm_props = []
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        glcm_props.extend(graycoprops(glcm, prop).astype(np.float32).ravel().tolist())
    feats.append(np.asarray(glcm_props, dtype=np.float32))
    return np.concatenate(feats).astype(np.float32)


def edge_descriptor(
    strip_img: np.ndarray,
    feature_mode: str = "fusion",
    stain_normalizer: StainNormalizer | None = None,
) -> np.ndarray:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    if feature_mode == "color":
        return color_descriptor(strip_img, stain_normalizer=stain_normalizer)
    if feature_mode == "texture":
        return texture_descriptor(strip_img, stain_normalizer=stain_normalizer)
    if feature_mode == "stain":
        return stain_descriptor(strip_img, stain_normalizer=stain_normalizer)
    return np.concatenate(
        [
            color_descriptor(strip_img, stain_normalizer=stain_normalizer),
            stain_descriptor(strip_img, stain_normalizer=stain_normalizer),
            texture_descriptor(strip_img, stain_normalizer=stain_normalizer),
        ]
    ).astype(np.float32)


def boundary_continuity_feature(strip_a: np.ndarray, strip_b: np.ndarray) -> np.ndarray:
    if strip_a.shape[0] >= strip_a.shape[1]:
        edge_a = normalize_image(strip_a[:, -1, :])
        edge_b = normalize_image(strip_b[:, 0, :])
    else:
        edge_a = normalize_image(strip_a[-1, :, :])
        edge_b = normalize_image(strip_b[0, :, :])

    if edge_a.shape[0] != edge_b.shape[0]:
        target_len = min(edge_a.shape[0], edge_b.shape[0])
        edge_a = cv2.resize(edge_a[np.newaxis, :, :], (target_len, 1), interpolation=cv2.INTER_LINEAR)[0]
        edge_b = cv2.resize(edge_b[np.newaxis, :, :], (target_len, 1), interpolation=cv2.INTER_LINEAR)[0]

    diff = np.abs(edge_a - edge_b)
    gray_a = edge_a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray_b = edge_b @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grad_gap = np.abs(np.diff(gray_a, prepend=gray_a[:1]) - np.diff(gray_b, prepend=gray_b[:1]))
    return np.array(
        [
            float(diff.mean()),
            float(diff.std()),
            float(np.linalg.norm(diff, axis=1).mean()),
            float(np.abs(gray_a.mean() - gray_b.mean())),
            float(grad_gap.mean()),
        ],
        dtype=np.float32,
    )


def pair_feature(
    strip_a: np.ndarray,
    strip_b: np.ndarray,
    feature_mode: str = "fusion",
    stain_normalizer: StainNormalizer | None = None,
) -> np.ndarray:
    da = edge_descriptor(strip_a, feature_mode=feature_mode, stain_normalizer=stain_normalizer)
    db = edge_descriptor(strip_b, feature_mode=feature_mode, stain_normalizer=stain_normalizer)
    abs_diff = np.abs(da - db)
    prod = da * db
    cosine = np.array(
        [np.dot(da, db) / ((np.linalg.norm(da) * np.linalg.norm(db)) + 1e-8)],
        dtype=np.float32
    )
    continuity = boundary_continuity_feature(strip_a, strip_b)
    return np.concatenate([abs_diff, prod, cosine, continuity]).astype(np.float32)


def tile_descriptor(
    tile_img: np.ndarray,
    feature_mode: str = "fusion",
    stain_normalizer: StainNormalizer | None = None,
) -> np.ndarray:
    resized = cv2.resize(tile_img, (64, 64), interpolation=cv2.INTER_AREA)
    return edge_descriptor(resized, feature_mode=feature_mode, stain_normalizer=stain_normalizer)


def state_one_hot(state: int, state_values: Sequence[int]) -> np.ndarray:
    encoded = np.zeros((len(state_values),), dtype=np.float32)
    encoded[state_values.index(state)] = 1.0
    return encoded


def stable_path_seed(path: str | Path) -> int:
    seed = SEED
    for ch in str(Path(path).resolve()):
        seed = ((seed * 131) + ord(ch)) % (2 ** 32)
    return seed

@dataclass
class PairSample:
    tile_a: np.ndarray
    tile_b: np.ndarray
    relation: str
    label: int


@dataclass(frozen=True)
class EdgeScoreComponents:
    rf: float
    nn: float
    fused: float


def coord_to_index(r: int, c: int, cols: int) -> int:
    return r * cols + c


def index_to_coord(idx: int, cols: int) -> Tuple[int, int]:
    return idx // cols, idx % cols


def are_neighbors(idx_a: int, idx_b: int, rows: int, cols: int, relation: str) -> bool:
    ra, ca = index_to_coord(idx_a, cols)
    rb, cb = index_to_coord(idx_b, cols)
    if relation == "right":
        return ra == rb and cb == ca + 1
    if relation == "down":
        return ca == cb and rb == ra + 1
    return False


def build_pair_samples_from_image(
    img: np.ndarray,
    rows: int,
    cols: int,
    negatives_per_positive: int = 2,
) -> List[PairSample]:
    tiles = split_into_tiles(img, rows, cols)
    return build_pair_samples_from_tiles(tiles, rows, cols, negatives_per_positive)


def build_pair_samples_from_tiles(
    tiles: Sequence[np.ndarray],
    rows: int,
    cols: int,
    negatives_per_positive: int = 2,
) -> List[PairSample]:
    tiles = list(tiles)
    n = len(tiles)
    pairs: List[PairSample] = []

    positives = []

    for r in range(rows):
        for c in range(cols - 1):
            a = coord_to_index(r, c, cols)
            b = coord_to_index(r, c + 1, cols)
            positives.append((a, b, "right"))

    for r in range(rows - 1):
        for c in range(cols):
            a = coord_to_index(r, c, cols)
            b = coord_to_index(r + 1, c, cols)
            positives.append((a, b, "down"))

    def hard_negative_targets(anchor_idx: int, positive_target_idx: int, relation: str) -> List[int]:
        ra, ca = index_to_coord(anchor_idx, cols)
        candidates: List[int] = []
        if relation == "right":
            for c in range(cols):
                idx = coord_to_index(ra, c, cols)
                if idx != anchor_idx and idx != positive_target_idx and not are_neighbors(anchor_idx, idx, rows, cols, relation):
                    candidates.append(idx)
            for r in range(rows):
                idx = coord_to_index(r, ca, cols)
                if idx != anchor_idx and idx != positive_target_idx and not are_neighbors(anchor_idx, idx, rows, cols, relation):
                    candidates.append(idx)
        else:
            for r in range(rows):
                idx = coord_to_index(r, ca, cols)
                if idx != anchor_idx and idx != positive_target_idx and not are_neighbors(anchor_idx, idx, rows, cols, relation):
                    candidates.append(idx)
            for c in range(cols):
                idx = coord_to_index(ra, c, cols)
                if idx != anchor_idx and idx != positive_target_idx and not are_neighbors(anchor_idx, idx, rows, cols, relation):
                    candidates.append(idx)
        return list(dict.fromkeys(candidates))

    for a, b, rel in positives:
        pairs.append(PairSample(tiles[a], tiles[b], rel, 1))
        neg_candidates: List[Tuple[int, int]] = []

        anchor_hard = hard_negative_targets(a, b, rel)
        if anchor_hard:
            neg_candidates.append((a, random.choice(anchor_hard)))

        reverse_hard = [idx for idx in range(n) if idx != a and idx != b and not are_neighbors(idx, b, rows, cols, rel)]
        if reverse_hard:
            neg_candidates.append((random.choice(reverse_hard), b))

        while len(neg_candidates) < negatives_per_positive:
            ia = random.randrange(n)
            ib = random.randrange(n)
            if ia != ib and not are_neighbors(ia, ib, rows, cols, rel):
                neg_candidates.append((ia, ib))

        for ia, ib in neg_candidates[:negatives_per_positive]:
            pairs.append(PairSample(tiles[ia], tiles[ib], rel, 0))
    return pairs


def build_dataset_from_image_paths(
    image_paths: Sequence[str | Path],
    rows: int,
    cols: int,
    negatives_per_positive: int = 2,
) -> List[PairSample]:
    if not image_paths:
        raise ValueError("At least one training image is required.")
    all_pairs = []
    for p in image_paths:
        img = read_image(p)
        all_pairs.extend(build_pair_samples_from_image(img, rows, cols, negatives_per_positive))
    return all_pairs


def build_pair_dataset(
    image_paths: Sequence[str | Path],
    rows: int,
    cols: int,
    negatives_per_positive: int,
    use_existing_slices: bool,
    slice_root: str | Path | None,
) -> List[PairSample]:
    if not image_paths:
        return []
    if use_existing_slices:
        return build_dataset_from_slice_folders(
            image_paths,
            rows=rows,
            cols=cols,
            negatives_per_positive=negatives_per_positive,
            slice_root=slice_root,
        )
    return build_dataset_from_image_paths(
        image_paths,
        rows=rows,
        cols=cols,
        negatives_per_positive=negatives_per_positive,
    )


def build_dataset_from_slice_folders(
    image_paths: Sequence[str | Path],
    rows: int,
    cols: int,
    negatives_per_positive: int = 2,
    slice_root: str | Path | None = None,
) -> List[PairSample]:
    if not image_paths:
        raise ValueError("At least one training image is required.")

    all_pairs = []
    missing_folders = []
    for image_path in image_paths:
        folder = find_slice_folder_for_image(image_path, rows, cols, slice_root=slice_root)
        if folder is None:
            missing_folders.append(str(image_path))
            continue
        tiles = load_tiles_from_folder(folder, rows=rows, cols=cols)
        all_pairs.extend(build_pair_samples_from_tiles(tiles, rows, cols, negatives_per_positive))

    if missing_folders:
        raise FileNotFoundError(
            "Missing slice folders for: " + ", ".join(missing_folders)
        )
    return all_pairs


def split_pair_samples(pair_samples: Sequence[PairSample], train_ratio: float = 0.8) -> Tuple[List[PairSample], List[PairSample]]:
    pair_samples = list(pair_samples)
    if len(pair_samples) < 2:
        raise ValueError("Need at least 2 pair samples to create train/validation splits.")

    split_idx = int(len(pair_samples) * train_ratio)
    split_idx = max(1, min(len(pair_samples) - 1, split_idx))
    return pair_samples[:split_idx], pair_samples[split_idx:]


def split_image_paths_for_validation(
    image_paths: Sequence[str | Path],
    val_ratio: float = 0.25,
) -> Tuple[List[Path], List[Path], str]:
    resolved_paths = [Path(p).resolve() for p in image_paths]
    if len(resolved_paths) <= 1:
        return resolved_paths, [], "pair_level_fallback"

    shuffled = list(resolved_paths)
    rng = random.Random(SEED)
    rng.shuffle(shuffled)

    val_count = int(round(len(shuffled) * val_ratio))
    val_count = max(1, min(len(shuffled) - 1, val_count))
    val_images = sorted(shuffled[:val_count])
    train_images = sorted(shuffled[val_count:])
    return train_images, val_images, "image_level"


def build_pair_sets_for_experiment(
    train_images: Sequence[str | Path],
    rows: int,
    cols: int,
    negatives_per_positive: int,
    use_existing_slices: bool,
    slice_root: str | Path | None,
    val_ratio: float,
) -> Tuple[List[PairSample], List[PairSample], Dict[str, object]]:
    train_split_images, val_split_images, split_strategy = split_image_paths_for_validation(
        train_images,
        val_ratio=val_ratio,
    )
    data_source = "full_images"

    if use_existing_slices:
        try:
            train_pairs = build_pair_dataset(
                train_split_images,
                rows,
                cols,
                negatives_per_positive,
                True,
                slice_root,
            )
            val_pairs = build_pair_dataset(
                val_split_images,
                rows,
                cols,
                negatives_per_positive,
                True,
                slice_root,
            )
            data_source = "existing_slice_folders"
        except FileNotFoundError as exc:
            print(f"Slice-folder dataset unavailable ({exc}); falling back to full images.")
            train_pairs = build_pair_dataset(
                train_split_images,
                rows,
                cols,
                negatives_per_positive,
                False,
                slice_root,
            )
            val_pairs = build_pair_dataset(
                val_split_images,
                rows,
                cols,
                negatives_per_positive,
                False,
                slice_root,
            )
    else:
        train_pairs = build_pair_dataset(
            train_split_images,
            rows,
            cols,
            negatives_per_positive,
            False,
            slice_root,
        )
        val_pairs = build_pair_dataset(
            val_split_images,
            rows,
            cols,
            negatives_per_positive,
            False,
            slice_root,
        )

    if not val_pairs:
        combined_pairs = train_pairs if train_pairs else build_pair_dataset(
            train_images,
            rows,
            cols,
            negatives_per_positive,
            False,
            slice_root,
        )
        train_pairs, val_pairs = split_pair_samples(combined_pairs)
        split_strategy = "pair_level_fallback"
        train_split_images = [Path(p).resolve() for p in train_images]
        val_split_images = []

    metadata = {
        "data_source": data_source,
        "split_strategy": split_strategy,
        "train_split_images": [str(p) for p in train_split_images],
        "val_split_images": [str(p) for p in val_split_images],
    }
    return train_pairs, val_pairs, metadata


def resolve_runtime_inputs(train_dir: str | None, test_image: str | None) -> Tuple[List[Path], Path]:
    base_dir = Path(train_dir) if train_dir else get_default_image_dir()
    train_images = list_image_paths(base_dir)

    if test_image:
        resolved_test = Path(test_image)
    else:
        resolved_test = train_images[-1]
        print(f"No --test_image provided; using {resolved_test.resolve()}")

    resolved_test = resolved_test.resolve()
    filtered_train_images = [p for p in train_images if p.resolve() != resolved_test]

    if filtered_train_images:
        return filtered_train_images, resolved_test

    print("Only one image was available, so the test image will also be used for training.")
    return train_images, resolved_test


def limit_image_paths(image_paths: Sequence[str | Path], limit: int) -> List[str | Path]:
    if limit <= 0:
        return list(image_paths)
    return list(image_paths)[:limit]

def prepare_ml_arrays(
    pair_samples: Sequence[PairSample],
    strip_size: int = 16,
    feature_mode: str = "fusion",
    stain_normalizer: StainNormalizer | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for s in pair_samples:
        strip_a, strip_b, _ = pair_edge_strips(s.tile_a, s.tile_b, s.relation, strip_size)
        xs.append(
            pair_feature(
                strip_a,
                strip_b,
                feature_mode=feature_mode,
                stain_normalizer=stain_normalizer,
            )
        )
        ys.append(s.label)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def train_random_forest(x_train: np.ndarray, y_train: np.ndarray, n_jobs: int = 1) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        random_state=SEED,
        n_jobs=n_jobs,
        class_weight="balanced"
    )
    clf.fit(x_train, y_train)
    return clf

class EdgePairDataset(Dataset):
    def __init__(
        self,
        pair_samples: Sequence[PairSample],
        strip_size: int = 16,
        input_mode: str = "rgb_hed",
        stain_normalizer: StainNormalizer | None = None,
    ):
        self.samples = list(pair_samples)
        self.strip_size = strip_size
        self.input_mode = input_mode
        self.stain_normalizer = stain_normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        strip_a, strip_b, relation_id = pair_edge_strips(s.tile_a, s.tile_b, s.relation, self.strip_size)
        strip_a = torch.from_numpy(
            strip_to_tensor(
                strip_a,
                input_mode=self.input_mode,
                stain_normalizer=self.stain_normalizer,
            )
        )
        strip_b = torch.from_numpy(
            strip_to_tensor(
                strip_b,
                input_mode=self.input_mode,
                stain_normalizer=self.stain_normalizer,
            )
        )
        y = torch.tensor([s.label], dtype=torch.float32)
        relation = torch.tensor(relation_id, dtype=torch.long)
        return strip_a, strip_b, relation, y


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + residual)
        return x


class EdgeEncoder(nn.Module):
    def __init__(self, input_channels: int = 3, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.layers = nn.Sequential(
            ResidualConvBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualConvBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualConvBlock(64, 96),
            ResidualConvBlock(96, 128),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, emb_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        x = x.flatten(1)
        return self.fc(x)


class PairNet(nn.Module):
    def __init__(self, emb_dim: int = 128, input_mode: str = "rgb_hed"):
        super().__init__()
        self.input_mode = input_mode
        input_channels = 3 if input_mode == "rgb" else 5
        self.encoder = EdgeEncoder(input_channels=input_channels, emb_dim=emb_dim)
        self.relation_emb = nn.Embedding(2, 16)
        self.cls = nn.Sequential(
            nn.Linear(emb_dim * 4 + 16, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, xa, xb, relation_ids):
        ea = self.encoder(xa)
        eb = self.encoder(xb)
        rel = self.relation_emb(relation_ids)
        feat = torch.cat([ea, eb, torch.abs(ea - eb), ea * eb, rel], dim=1)
        return self.cls(feat)


def train_pairnet(
    train_pairs: Sequence[PairSample],
    val_pairs: Sequence[PairSample],
    strip_size: int = 16,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    input_mode: str = "rgb_hed",
    model: PairNet | None = None,
    stain_normalizer: StainNormalizer | None = None,
) -> PairNet:
    train_ds = EdgePairDataset(
        train_pairs,
        strip_size,
        input_mode=input_mode,
        stain_normalizer=stain_normalizer,
    )
    val_ds = EdgePairDataset(
        val_pairs,
        strip_size,
        input_mode=input_mode,
        stain_normalizer=stain_normalizer,
    )
    pin_memory = device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    if model is None:
        model = PairNet(input_mode=input_mode).to(device)
    else:
        if model.input_mode != input_mode:
            raise ValueError(f"Expected warm-start PairNet input_mode={input_mode}, got {model.input_mode}")
        model = model.to(device)
    use_amp = device.startswith("cuda")
    amp_dtype = torch.float16
    scaler = make_grad_scaler(device, enabled=use_amp)
    pos_count = max(sum(1 for sample in train_pairs if sample.label == 1), 1)
    neg_count = max(sum(1 for sample in train_pairs if sample.label == 0), 1)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_key = (-1.0, -1.0, -1.0)
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xa, xb, relation_ids, y in train_loader:
            xa = xa.to(device, non_blocking=pin_memory)
            xb = xb.to(device, non_blocking=pin_memory)
            relation_ids = relation_ids.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            amp_ctx = amp_autocast(device, enabled=use_amp, dtype=amp_dtype)
            with amp_ctx:
                logits = model(xa, xb, relation_ids)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * xa.size(0)
        scheduler.step()

        model.eval()
        preds, gts, probs = [], [], []
        with torch.no_grad():
            for xa, xb, relation_ids, y in val_loader:
                xa = xa.to(device, non_blocking=pin_memory)
                xb = xb.to(device, non_blocking=pin_memory)
                relation_ids = relation_ids.to(device, non_blocking=pin_memory)
                amp_ctx = amp_autocast(device, enabled=use_amp, dtype=amp_dtype)
                with amp_ctx:
                    logits = model(xa, xb, relation_ids)
                prob = torch.sigmoid(logits).cpu().numpy().ravel()
                probs.extend(prob.tolist())
                preds.extend((prob >= 0.5).astype(np.int32).tolist())
                gts.extend(y.numpy().ravel().astype(np.int32).tolist())

        val_acc = accuracy_score(gts, preds)
        has_mixed_labels = len(set(gts)) > 1
        val_auc = roc_auc_score(gts, probs) if has_mixed_labels else 0.5
        val_ap = average_precision_score(gts, probs) if has_mixed_labels else float(np.mean(gts))
        avg_loss = running_loss / max(len(train_ds), 1)
        print(
            f"[Epoch {epoch:02d}] train_loss={avg_loss:.4f} "
            f"val_auc={val_auc:.4f} val_ap={val_ap:.4f} val_acc={val_acc:.4f}"
        )

        model_key = (val_auc, val_ap, val_acc)
        if model_key > best_key:
            best_key = model_key
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_pairnet_prob_batch(
    model: PairNet,
    strip_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    relation: str,
    device: str = "cpu",
    batch_size: int = 512,
    stain_normalizer: StainNormalizer | None = None,
) -> np.ndarray:
    if not strip_pairs:
        return np.empty((0,), dtype=np.float32)

    model.eval()
    all_probs = []
    relation_id = relation_to_id(relation)
    use_amp = str(device).startswith("cuda")
    amp_dtype = torch.float16
    with torch.no_grad():
        for start in range(0, len(strip_pairs), batch_size):
            batch = strip_pairs[start:start + batch_size]
            xa = np.stack(
                [
                    strip_to_tensor(
                        strip_a,
                        input_mode=model.input_mode,
                        stain_normalizer=stain_normalizer,
                    )
                    for strip_a, _ in batch
                ]
            ).astype(np.float32)
            xb = np.stack(
                [
                    strip_to_tensor(
                        strip_b,
                        input_mode=model.input_mode,
                        stain_normalizer=stain_normalizer,
                    )
                    for _, strip_b in batch
                ]
            ).astype(np.float32)
            xa_tensor = torch.from_numpy(xa).to(device)
            xb_tensor = torch.from_numpy(xb).to(device)
            relation_ids = torch.full(
                (len(batch),),
                relation_id,
                dtype=torch.long,
                device=device,
            )
            amp_ctx = amp_autocast(device, enabled=use_amp, dtype=amp_dtype)
            with amp_ctx:
                probs = torch.sigmoid(model(xa_tensor, xb_tensor, relation_ids)).cpu().numpy().ravel()
            all_probs.append(probs.astype(np.float32))
    return np.concatenate(all_probs, axis=0)

class FusionScorer:
    def __init__(
        self,
        rf_model: RandomForestClassifier | None,
        nn_model: PairNet | None,
        device: str = "cpu",
        alpha: float = 0.5,
        feature_mode: str = "fusion",
        stain_normalizer: StainNormalizer | None = None,
        graph_model: GraphEdgeReranker | None = None,
        graph_weight: float = 0.0,
        graph_candidate_k: int = 12,
    ):
        self.rf_model = rf_model
        self.nn_model = nn_model
        self.device = device
        self.alpha = min(max(alpha, 0.0), 1.0)
        self.feature_mode = feature_mode
        self.stain_normalizer = stain_normalizer
        self.graph_model = graph_model
        self.graph_weight = min(max(graph_weight, 0.0), 1.0)
        self.graph_candidate_k = max(int(graph_candidate_k), 1)
        self.use_rf = self.rf_model is not None and self.alpha > 0.0
        self.use_nn = self.nn_model is not None and self.alpha < 1.0
        self.use_graph = self.graph_model is not None and self.graph_weight > 0.0
        if not self.use_rf and not self.use_nn:
            raise ValueError("FusionScorer requires at least one enabled model branch.")

    def score_components_batch(
        self,
        strip_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
        relation: str,
        nn_batch_size: int = 512,
    ) -> Dict[str, np.ndarray]:
        if not strip_pairs:
            empty = np.empty((0,), dtype=np.float32)
            return {"rf": empty, "nn": empty, "fused": empty}

        rf_scores = np.full((len(strip_pairs),), 0.5, dtype=np.float32)
        nn_scores = np.full((len(strip_pairs),), 0.5, dtype=np.float32)
        if self.use_rf:
            ml_x = np.asarray(
                [
                    pair_feature(
                        strip_a,
                        strip_b,
                        feature_mode=self.feature_mode,
                        stain_normalizer=self.stain_normalizer,
                    )
                    for strip_a, strip_b in strip_pairs
                ],
                dtype=np.float32,
            )
            rf_scores = self.rf_model.predict_proba(ml_x)[:, 1].astype(np.float32)
        if self.use_nn:
            nn_scores = predict_pairnet_prob_batch(
                self.nn_model,
                strip_pairs,
                relation=relation,
                device=self.device,
                batch_size=nn_batch_size,
                stain_normalizer=self.stain_normalizer,
            )
        weights = []
        scores = []
        if self.use_rf:
            weights.append(self.alpha)
            scores.append(rf_scores)
        if self.use_nn:
            weights.append(1.0 - self.alpha)
            scores.append(nn_scores)
        if len(scores) == 1:
            fused_scores = scores[0]
        else:
            total_weight = sum(weights)
            fused_scores = sum(score * (weight / total_weight) for score, weight in zip(scores, weights))
        return {
            "rf": rf_scores.astype(np.float32),
            "nn": nn_scores.astype(np.float32),
            "fused": fused_scores.astype(np.float32),
        }

    def score_batch(
        self,
        strip_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
        relation: str,
        nn_batch_size: int = 512,
    ) -> np.ndarray:
        return self.score_components_batch(
            strip_pairs,
            relation=relation,
            nn_batch_size=nn_batch_size,
        )["fused"]

@dataclass
class Piece:
    piece_id: int
    image: np.ndarray
    gt_position: Tuple[int, int]
    scramble_state: int = 0


@dataclass
class GraphCandidateSample:
    node_features: np.ndarray
    piece_indices: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_base_scores: np.ndarray
    edge_labels: np.ndarray | None
    edge_keys: List[Tuple[int, int, int, int, str]]


def load_randomized_tiles_and_pieces(
    folder: str | Path,
    rows: int,
    cols: int,
) -> Tuple[List[np.ndarray], List[Piece]]:
    folder = Path(folder)
    metadata_rows = read_slice_metadata(folder)
    if metadata_rows is None:
        raise FileNotFoundError(f"Expected metadata.csv in randomized slice folder: {folder}")

    tile_paths = {path.name: path for path in list_image_paths(folder)}
    expected_count = rows * cols
    sorted_rows = sorted(metadata_rows, key=lambda row: int(row["output_index"]))
    if len(sorted_rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} metadata rows in {folder}, got {len(sorted_rows)}"
        )

    ordered_tiles: List[np.ndarray | None] = [None] * expected_count
    pieces: List[Piece] = []

    for row in sorted_rows:
        output_file = row["output_file"]
        tile_path = tile_paths.get(output_file)
        if tile_path is None:
            raise FileNotFoundError(f"Tile listed in metadata is missing: {folder / output_file}")

        original_row = int(row["original_row"]) - 1
        original_col = int(row["original_col"]) - 1
        expected_piece_id = coord_to_index(original_row, original_col, cols)
        original_index = int(row["original_index"]) - 1
        if original_index != expected_piece_id:
            raise ValueError(
                f"Metadata mismatch in {folder}: original_index={original_index + 1} "
                f"but row/col imply {expected_piece_id + 1}"
            )

        tile_image = read_image(tile_path)
        saved_state = orientation_state(
            int(row.get("rotation_degrees", 0)),
            parse_bool_like(row.get("horizontal_flip", 0)),
        )
        correction_state = inverse_state(saved_state)
        ordered_tiles[expected_piece_id] = apply_state(tile_image, correction_state)
        pieces.append(
            Piece(
                piece_id=expected_piece_id,
                image=tile_image,
                gt_position=(original_row, original_col),
                scramble_state=correction_state,
            )
        )

    if any(tile is None for tile in ordered_tiles):
        raise ValueError(f"Metadata in {folder} does not cover every tile position.")

    return [tile for tile in ordered_tiles if tile is not None], pieces


def scramble_tile_list(
    original_tiles: Sequence[np.ndarray],
    rows: int,
    cols: int,
    allow_flips: bool = True,
) -> Tuple[List[Piece], List[np.ndarray]]:
    original_tiles = list(original_tiles)
    if len(original_tiles) != rows * cols:
        raise ValueError(f"Expected {rows * cols} tiles, got {len(original_tiles)}")
    pieces = []
    for idx, tile in enumerate(original_tiles):
        r, c = index_to_coord(idx, cols)
        state = random.choice([0, 1, 2, 3]) if allow_flips else 0
        tile_scrambled = apply_state(tile, state)
        pieces.append(Piece(piece_id=idx, image=tile_scrambled, gt_position=(r, c), scramble_state=state))
    random.shuffle(pieces)
    return pieces, original_tiles


def build_identity_pieces_from_tiles(
    tiles: Sequence[np.ndarray],
    cols: int,
) -> List[Piece]:
    pieces = []
    for idx, tile in enumerate(tiles):
        pieces.append(
            Piece(
                piece_id=idx,
                image=tile.copy(),
                gt_position=index_to_coord(idx, cols),
                scramble_state=0,
            )
        )
    return pieces


def build_scrambled_graph_training_pieces(
    tiles: Sequence[np.ndarray],
    cols: int,
    state_values: Sequence[int],
    rng: random.Random,
) -> List[Piece]:
    resolved_states = tuple(state_values) if state_values else (0,)
    pieces = []
    for idx, tile in enumerate(tiles):
        applied_state = rng.choice(resolved_states)
        scrambled_tile = apply_state(tile, applied_state)
        correction_state = inverse_state(applied_state) if len(resolved_states) > 1 else 0
        pieces.append(
            Piece(
                piece_id=idx,
                image=scrambled_tile,
                gt_position=index_to_coord(idx, cols),
                scramble_state=correction_state,
            )
        )
    return pieces


def load_upright_tiles_for_image(
    image_path: str | Path,
    rows: int,
    cols: int,
    use_existing_slices: bool,
    slice_root: str | Path | None,
) -> List[np.ndarray]:
    if use_existing_slices:
        folder = find_slice_folder_for_image(image_path, rows, cols, slice_root=slice_root)
        if folder is not None:
            return load_tiles_from_folder(folder, rows=rows, cols=cols)
    return split_into_tiles(read_image(image_path), rows, cols)


def compute_base_score_components_lookup(
    pieces: Sequence[Piece],
    scorer: "FusionScorer",
    strip_size: int = 16,
    score_batch_size: int = 512,
    state_values: Sequence[int] = (0, 1, 2, 3),
    progress_label: str | None = None,
    progress_interval_seconds: float = 30.0,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int, int, int, str], EdgeScoreComponents]]:
    oriented_tiles: Dict[Tuple[int, int], np.ndarray] = {}
    edge_cache: Dict[Tuple[int, int, str], np.ndarray] = {}
    component_lookup: Dict[Tuple[int, int, int, int, str], EdgeScoreComponents] = {}

    for piece in pieces:
        for state in state_values:
            oriented_tile = apply_state(piece.image, state)
            oriented_tiles[(piece.piece_id, state)] = oriented_tile
            edge_cache[(piece.piece_id, state, "right")] = get_edge_strip(oriented_tile, "right", strip_size)
            edge_cache[(piece.piece_id, state, "left")] = get_edge_strip(oriented_tile, "left", strip_size)
            edge_cache[(piece.piece_id, state, "bottom")] = get_edge_strip(oriented_tile, "bottom", strip_size)
            edge_cache[(piece.piece_id, state, "top")] = get_edge_strip(oriented_tile, "top", strip_size)

    batch_keys: List[Tuple[int, int, int, int, str]] = []
    batch_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    relations = ("right", "down")
    piece_count = len(pieces)
    state_count = len(tuple(state_values))
    total_edge_scores = len(relations) * piece_count * max(piece_count - 1, 0) * (state_count ** 2)
    processed_edge_scores = 0
    progress_start = time.perf_counter()
    last_progress_time = progress_start

    def flush_batch(relation_name: str) -> None:
        nonlocal processed_edge_scores
        nonlocal last_progress_time
        if not batch_pairs:
            return
        batch_size = len(batch_pairs)
        components = scorer.score_components_batch(
            batch_pairs,
            relation=relation_name,
            nn_batch_size=score_batch_size,
        )
        for key, rf_score, nn_score, fused_score in zip(
            batch_keys,
            components["rf"],
            components["nn"],
            components["fused"],
        ):
            component_lookup[key] = EdgeScoreComponents(
                rf=float(rf_score),
                nn=float(nn_score),
                fused=float(fused_score),
            )
        batch_keys.clear()
        batch_pairs.clear()
        processed_edge_scores += batch_size
        if progress_label:
            now = time.perf_counter()
            should_report = processed_edge_scores >= total_edge_scores
            should_report = should_report or ((now - last_progress_time) >= progress_interval_seconds)
            if should_report:
                progress_fraction = 0.0 if total_edge_scores <= 0 else (processed_edge_scores / total_edge_scores)
                print_status(
                    f"[{progress_label}] edge_scores={processed_edge_scores}/{total_edge_scores} "
                    f"({progress_fraction:.1%}) elapsed={format_elapsed(now - progress_start)}"
                )
                last_progress_time = now

    piece_ids = [piece.piece_id for piece in pieces]
    for relation in relations:
        src_side = "right" if relation == "right" else "bottom"
        dst_side = "left" if relation == "right" else "top"
        for a_id in piece_ids:
            for b_id in piece_ids:
                if a_id == b_id:
                    continue
                for a_state in state_values:
                    strip_a = edge_cache[(a_id, a_state, src_side)]
                    for b_state in state_values:
                        strip_b = edge_cache[(b_id, b_state, dst_side)]
                        batch_keys.append((a_id, a_state, b_id, b_state, relation))
                        batch_pairs.append((strip_a, strip_b))
                        if len(batch_pairs) >= score_batch_size:
                            flush_batch(relation)
        flush_batch(relation)
    return oriented_tiles, component_lookup


def build_graph_candidate_sample(
    pieces: Sequence[Piece],
    rows: int | None,
    cols: int | None,
    oriented_tiles: Dict[Tuple[int, int], np.ndarray],
    component_lookup: Dict[Tuple[int, int, int, int, str], EdgeScoreComponents],
    state_values: Sequence[int],
    feature_mode: str = "fusion",
    stain_normalizer: StainNormalizer | None = None,
    candidate_k: int = 12,
    include_labels: bool = False,
) -> GraphCandidateSample:
    resolved_state_values = list(state_values) if state_values else [0]
    node_features: List[np.ndarray] = []
    piece_indices: List[int] = []
    node_lookup: Dict[Tuple[int, int], int] = {}

    for piece in pieces:
        for state in resolved_state_values:
            node_lookup[(piece.piece_id, state)] = len(node_features)
            node_features.append(
                np.concatenate(
                    [
                        tile_descriptor(
                            oriented_tiles[(piece.piece_id, state)],
                            feature_mode=feature_mode,
                            stain_normalizer=stain_normalizer,
                        ),
                        state_one_hot(state, resolved_state_values),
                    ]
                ).astype(np.float32)
            )
            piece_indices.append(piece.piece_id)

    piece_lookup = {piece.piece_id: piece for piece in pieces}
    grouped_candidates: Dict[Tuple[int, str], List[Tuple[float, Tuple[int, int, int, int, str], int, np.ndarray, int]]] = defaultdict(list)

    for key, components in component_lookup.items():
        a_id, a_state, b_id, b_state, relation = key
        src_idx = node_lookup[(a_id, a_state)]
        dst_idx = node_lookup[(b_id, b_state)]
        relation_one_hot = np.zeros((2,), dtype=np.float32)
        relation_one_hot[relation_to_id(relation)] = 1.0
        edge_feature = np.concatenate(
            [
                np.array(
                    [
                        components.rf,
                        components.nn,
                        components.fused,
                        abs(components.rf - components.nn),
                    ],
                    dtype=np.float32,
                ),
                relation_one_hot,
            ]
        ).astype(np.float32)
        label = 0
        if include_labels:
            if rows is None or cols is None:
                raise ValueError("rows and cols are required when include_labels=True.")
            label = int(
                are_neighbors(a_id, b_id, rows, cols, relation)
                and a_state == piece_lookup[a_id].scramble_state
                and b_state == piece_lookup[b_id].scramble_state
            )
        grouped_candidates[(src_idx, relation)].append((components.fused, key, dst_idx, edge_feature, label))

    selected_records: List[Tuple[Tuple[int, int, int, int, str], int, np.ndarray, int]] = []
    selected_keys = set()
    keep_count = max(int(candidate_k), 1)
    for records in grouped_candidates.values():
        ranked = sorted(records, key=lambda item: item[0], reverse=True)
        for _, key, dst_idx, edge_feature, label in ranked[:keep_count]:
            selected_records.append((key, dst_idx, edge_feature, label))
            selected_keys.add(key)

    if include_labels:
        for key, components in component_lookup.items():
            a_id, a_state, b_id, b_state, relation = key
            if rows is None or cols is None:
                raise ValueError("rows and cols are required when include_labels=True.")
            label = int(
                are_neighbors(a_id, b_id, rows, cols, relation)
                and a_state == piece_lookup[a_id].scramble_state
                and b_state == piece_lookup[b_id].scramble_state
            )
            if label != 1 or key in selected_keys:
                continue
            relation_one_hot = np.zeros((2,), dtype=np.float32)
            relation_one_hot[relation_to_id(relation)] = 1.0
            edge_feature = np.concatenate(
                [
                    np.array(
                        [
                            components.rf,
                            components.nn,
                            components.fused,
                            abs(components.rf - components.nn),
                        ],
                        dtype=np.float32,
                    ),
                    relation_one_hot,
                ]
            ).astype(np.float32)
            selected_records.append((key, node_lookup[(b_id, b_state)], edge_feature, label))
            selected_keys.add(key)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_features: List[np.ndarray] = []
    edge_base_scores: List[float] = []
    edge_labels: List[float] = []
    edge_keys: List[Tuple[int, int, int, int, str]] = []

    for key, dst_idx, edge_feature, label in selected_records:
        a_id, a_state, _, _, _ = key
        edge_src.append(node_lookup[(a_id, a_state)])
        edge_dst.append(dst_idx)
        edge_features.append(edge_feature)
        edge_base_scores.append(component_lookup[key].fused)
        edge_keys.append(key)
        if include_labels:
            edge_labels.append(float(label))

    if not edge_features:
        raise ValueError("Graph candidate sample has no selected edges.")

    return GraphCandidateSample(
        node_features=np.asarray(node_features, dtype=np.float32),
        piece_indices=np.asarray(piece_indices, dtype=np.int64),
        edge_index=np.asarray([edge_src, edge_dst], dtype=np.int64),
        edge_features=np.asarray(edge_features, dtype=np.float32),
        edge_base_scores=np.asarray(edge_base_scores, dtype=np.float32),
        edge_labels=np.asarray(edge_labels, dtype=np.float32) if include_labels else None,
        edge_keys=edge_keys,
    )


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.shape[-1]))
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.index_add_(0, index, src)
    counts = src.new_zeros((dim_size, 1))
    counts.index_add_(0, index, torch.ones((index.shape[0], 1), device=src.device, dtype=src.dtype))
    return out / counts.clamp_min(1.0)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.shape[-1]))
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.index_add_(0, index, src)
    return out


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    weights = torch.zeros_like(scores)
    for idx in range(dim_size):
        mask = index == idx
        if mask.any():
            weights[mask] = torch.softmax(scores[mask], dim=0)
    return weights


class GraphMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        piece_indices: torch.Tensor,
        num_pieces: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        messages = self.message_mlp(torch.cat([node_x[src], node_x[dst], edge_attr], dim=1))
        agg_out = scatter_mean(messages, src, node_x.shape[0])
        agg_in = scatter_mean(messages, dst, node_x.shape[0])
        piece_context = scatter_mean(node_x, piece_indices, num_pieces)[piece_indices]
        node_delta = self.node_mlp(torch.cat([node_x, agg_in, agg_out, piece_context], dim=1))
        node_x = self.node_norm(node_x + node_delta)
        edge_delta = self.edge_mlp(torch.cat([edge_attr, node_x[src], node_x[dst]], dim=1))
        edge_attr = self.edge_norm(edge_attr + edge_delta)
        return node_x, edge_attr


class GraphAttentionMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_score = nn.Linear(hidden_dim * 3, 1)
        self.in_score = nn.Linear(hidden_dim * 3, 1)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        piece_indices: torch.Tensor,
        num_pieces: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        triplet = torch.cat([node_x[src], node_x[dst], edge_attr], dim=1)
        messages = self.message_mlp(triplet)
        out_weights = scatter_softmax(self.act(self.out_score(triplet)).squeeze(1), src, node_x.shape[0]).unsqueeze(1)
        in_weights = scatter_softmax(self.act(self.in_score(triplet)).squeeze(1), dst, node_x.shape[0]).unsqueeze(1)
        agg_out = scatter_sum(messages * out_weights, src, node_x.shape[0])
        agg_in = scatter_sum(messages * in_weights, dst, node_x.shape[0])
        piece_context = scatter_mean(node_x, piece_indices, num_pieces)[piece_indices]
        node_delta = self.node_mlp(torch.cat([node_x, agg_in, agg_out, piece_context], dim=1))
        node_x = self.node_norm(node_x + node_delta)
        edge_delta = self.edge_mlp(torch.cat([edge_attr, node_x[src], node_x[dst]], dim=1))
        edge_attr = self.edge_norm(edge_attr + edge_delta)
        return node_x, edge_attr


class GraphEdgeReranker(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        model_type: str = "graphsage",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.model_type = model_type
        if model_type not in {"graphsage", "gat"}:
            raise ValueError(f"Unsupported graph model_type: {model_type}")
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        layer_cls = GraphMessagePassingLayer if model_type == "graphsage" else GraphAttentionMessagePassingLayer
        self.layers = nn.ModuleList([layer_cls(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        piece_indices: torch.Tensor,
    ) -> torch.Tensor:
        if edge_features.shape[0] == 0:
            return edge_features.new_zeros((0,))
        node_x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)
        num_pieces = int(piece_indices.max().item()) + 1 if piece_indices.numel() else 0
        for layer in self.layers:
            node_x, edge_attr = layer(node_x, edge_attr, edge_index, piece_indices, num_pieces)
        src, dst = edge_index
        logits = self.classifier(torch.cat([edge_attr, node_x[src], node_x[dst]], dim=1))
        return logits.squeeze(1)


def graph_sample_to_tensors(
    sample: GraphCandidateSample,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    node_features = torch.from_numpy(sample.node_features).to(device)
    piece_indices = torch.from_numpy(sample.piece_indices).to(device)
    edge_index = torch.from_numpy(sample.edge_index).to(device)
    edge_features = torch.from_numpy(sample.edge_features).to(device)
    edge_base_scores = torch.from_numpy(sample.edge_base_scores).to(device)
    edge_labels = (
        torch.from_numpy(sample.edge_labels).to(device)
        if sample.edge_labels is not None
        else None
    )
    return node_features, piece_indices, edge_index, edge_features, edge_base_scores, edge_labels


def compute_graph_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    base_scores: torch.Tensor,
    hard_negative_ratio: float = 8.0,
) -> torch.Tensor:
    positive_indices = torch.nonzero(labels > 0.5, as_tuple=False).squeeze(1)
    negative_indices = torch.nonzero(labels <= 0.5, as_tuple=False).squeeze(1)
    if positive_indices.numel() == 0 or negative_indices.numel() == 0:
        return F.binary_cross_entropy_with_logits(logits, labels)

    max_negatives = max(int(math.ceil(positive_indices.numel() * hard_negative_ratio)), positive_indices.numel())
    if negative_indices.numel() > max_negatives:
        hard_neg_idx = torch.topk(base_scores[negative_indices], k=max_negatives, largest=True).indices
        negative_indices = negative_indices[hard_neg_idx]
    selected = torch.cat([positive_indices, negative_indices], dim=0)
    selected_logits = logits[selected]
    selected_labels = labels[selected]
    pos_count = max(int((selected_labels > 0.5).sum().item()), 1)
    neg_count = max(int((selected_labels <= 0.5).sum().item()), 1)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(selected_logits, selected_labels, pos_weight=pos_weight)


def evaluate_graph_reranker(
    model: GraphEdgeReranker,
    graph_samples: Sequence[GraphCandidateSample],
    device: str,
) -> Dict[str, float]:
    if not graph_samples:
        return {"auc": 0.5, "ap": 0.0, "acc": 0.0}

    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    use_amp = str(device).startswith("cuda") and getattr(model, "model_type", "graphsage") != "gat"
    amp_dtype = torch.float16
    eval_start = time.perf_counter()
    last_progress_time = eval_start
    total_samples = len(graph_samples)
    with torch.no_grad():
        for sample_index, sample in enumerate(graph_samples, start=1):
            node_features, piece_indices, edge_index, edge_features, _, edge_labels = graph_sample_to_tensors(sample, device)
            if edge_labels is None:
                continue
            amp_ctx = amp_autocast(device, enabled=use_amp, dtype=amp_dtype)
            with amp_ctx:
                logits = model(node_features, edge_index, edge_features, piece_indices)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs.tolist())
            all_labels.extend(edge_labels.cpu().numpy().astype(np.int32).ravel().tolist())
            now = time.perf_counter()
            if total_samples > 1 and ((now - last_progress_time) >= 30.0 or sample_index == total_samples):
                print_status(
                    f"[Graph Eval] samples={sample_index}/{total_samples} "
                    f"elapsed={format_elapsed(now - eval_start)}"
                )
                last_progress_time = now

    if not all_labels:
        return {"auc": 0.5, "ap": 0.0, "acc": 0.0}

    has_mixed_labels = len(set(all_labels)) > 1
    auc = roc_auc_score(all_labels, all_probs) if has_mixed_labels else 0.5
    ap = average_precision_score(all_labels, all_probs) if has_mixed_labels else float(np.mean(all_labels))
    acc = accuracy_score(all_labels, (np.asarray(all_probs) >= 0.5).astype(np.int32))
    return {"auc": float(auc), "ap": float(ap), "acc": float(acc)}


def train_graph_reranker(
    train_graphs: Sequence[GraphCandidateSample],
    val_graphs: Sequence[GraphCandidateSample],
    device: str,
    hidden_dim: int = 128,
    num_layers: int = 3,
    epochs: int = 8,
    lr: float = 1e-3,
    hard_negative_ratio: float = 8.0,
    model_type: str = "graphsage",
) -> Tuple[GraphEdgeReranker, Dict[str, float]]:
    if not train_graphs:
        raise ValueError("At least one graph sample is required to train the graph reranker.")

    first_graph = train_graphs[0]
    model = GraphEdgeReranker(
        node_input_dim=first_graph.node_features.shape[1],
        edge_input_dim=first_graph.edge_features.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        model_type=model_type,
    ).to(device)
    use_amp = device.startswith("cuda") and model_type != "gat"
    amp_dtype = torch.float16
    scaler = make_grad_scaler(device, enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_key = (-1.0, -1.0, -1.0)
    best_state = None
    best_metrics = {"auc": 0.5, "ap": 0.0, "acc": 0.0}
    print_status(
        f"[Graph Training] model={model_type} train_graphs={len(train_graphs)} "
        f"val_graphs={len(val_graphs)} epochs={epochs}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        order = list(range(len(train_graphs)))
        random.Random(SEED + epoch).shuffle(order)
        epoch_start = time.perf_counter()
        last_progress_time = epoch_start

        for sample_index, idx in enumerate(order, start=1):
            sample = train_graphs[idx]
            node_features, piece_indices, edge_index, edge_features, edge_base_scores, edge_labels = graph_sample_to_tensors(sample, device)
            if edge_labels is None:
                continue
            optimizer.zero_grad()
            amp_ctx = amp_autocast(device, enabled=use_amp, dtype=amp_dtype)
            with amp_ctx:
                logits = model(node_features, edge_index, edge_features, piece_indices)
                loss = compute_graph_training_loss(
                    logits,
                    edge_labels,
                    edge_base_scores,
                    hard_negative_ratio=hard_negative_ratio,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())
            now = time.perf_counter()
            if len(order) > 1 and ((now - last_progress_time) >= 30.0 or sample_index == len(order)):
                avg_loss_so_far = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                print_status(
                    f"[Graph Epoch {epoch:02d}] train_samples={sample_index}/{len(order)} "
                    f"avg_loss={avg_loss_so_far:.4f} elapsed={format_elapsed(now - epoch_start)}"
                )
                last_progress_time = now

        scheduler.step()
        val_metrics = evaluate_graph_reranker(model, val_graphs, device=device)
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print_status(
            f"[Graph Epoch {epoch:02d}] train_loss={avg_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_ap={val_metrics['ap']:.4f} val_acc={val_metrics['acc']:.4f}"
        )
        model_key = (val_metrics["auc"], val_metrics["ap"], val_metrics["acc"])
        if model_key > best_key:
            best_key = model_key
            best_metrics = val_metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    best_metrics = {
        **best_metrics,
        "model_type": model_type,
        "hidden_dim": float(hidden_dim),
        "num_layers": float(num_layers),
        "epochs": float(epochs),
    }
    return model, best_metrics


def predict_graph_reranker_probs(
    model: GraphEdgeReranker,
    graph_sample: GraphCandidateSample,
    device: str,
) -> np.ndarray:
    model.eval()
    use_amp = str(device).startswith("cuda") and getattr(model, "model_type", "graphsage") != "gat"
    amp_dtype = torch.float16
    with torch.no_grad():
        node_features, piece_indices, edge_index, edge_features, _, _ = graph_sample_to_tensors(graph_sample, device)
        amp_ctx = amp_autocast(device, enabled=use_amp, dtype=amp_dtype)
        with amp_ctx:
            logits = model(node_features, edge_index, edge_features, piece_indices)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32).ravel()


def build_graph_samples_from_image_paths(
    image_paths: Sequence[str | Path],
    rows: int,
    cols: int,
    scorer: FusionScorer,
    use_existing_slices: bool,
    slice_root: str | Path | None,
    strip_size: int,
    state_values: Sequence[int],
    candidate_k: int,
) -> List[GraphCandidateSample]:
    graph_samples = []
    resolved_states = tuple(state_values) if state_values else (0,)
    total_images = len(image_paths)
    build_start = time.perf_counter()
    if total_images:
        print_status(
            f"[Graph Sample Build] images={total_images} rows={rows} cols={cols} "
            f"states={len(resolved_states)} candidate_k={candidate_k}"
        )
    for image_index, image_path in enumerate(image_paths, start=1):
        image_name = Path(image_path).name
        print_status(f"[Graph Sample Build] image={image_index}/{total_images} start={image_name}")
        tiles = load_upright_tiles_for_image(
            image_path,
            rows=rows,
            cols=cols,
            use_existing_slices=use_existing_slices,
            slice_root=slice_root,
        )
        if len(resolved_states) == 1 and resolved_states[0] == 0:
            pieces = build_identity_pieces_from_tiles(tiles, cols=cols)
        else:
            rng = random.Random(stable_path_seed(image_path))
            pieces = build_scrambled_graph_training_pieces(
                tiles,
                cols=cols,
                state_values=resolved_states,
                rng=rng,
            )
        oriented_tiles, component_lookup = compute_base_score_components_lookup(
            pieces,
            scorer,
            strip_size=strip_size,
            state_values=resolved_states,
            progress_label=f"Graph Sample Build {image_index}/{total_images} {image_name}",
        )
        graph_sample = build_graph_candidate_sample(
            pieces,
            rows=rows,
            cols=cols,
            oriented_tiles=oriented_tiles,
            component_lookup=component_lookup,
            state_values=resolved_states,
            feature_mode=scorer.feature_mode,
            stain_normalizer=scorer.stain_normalizer,
            candidate_k=candidate_k,
            include_labels=True,
        )
        graph_samples.append(graph_sample)
        print_status(
            f"[Graph Sample Build] image={image_index}/{total_images} done={image_name} "
            f"nodes={graph_sample.node_features.shape[0]} "
            f"edges={graph_sample.edge_features.shape[0]} "
            f"elapsed={format_elapsed(time.perf_counter() - build_start)}"
        )
    return graph_samples


def precompute_score_lookup(
    pieces: Sequence[Piece],
    scorer: FusionScorer,
    strip_size: int = 16,
    score_batch_size: int = 512,
    state_values: Sequence[int] = (0, 1, 2, 3),
    progress_label: str | None = None,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int, int, int, str], float]]:
    oriented_tiles, component_lookup = compute_base_score_components_lookup(
        pieces,
        scorer,
        strip_size=strip_size,
        score_batch_size=score_batch_size,
        state_values=state_values,
        progress_label=progress_label,
    )
    score_lookup = {
        key: components.fused
        for key, components in component_lookup.items()
    }
    if scorer.use_graph and scorer.graph_model is not None:
        graph_sample = build_graph_candidate_sample(
            pieces,
            rows=None,
            cols=None,
            oriented_tiles=oriented_tiles,
            component_lookup=component_lookup,
            state_values=state_values,
            feature_mode=scorer.feature_mode,
            stain_normalizer=scorer.stain_normalizer,
            candidate_k=scorer.graph_candidate_k,
            include_labels=False,
        )
        graph_probs = predict_graph_reranker_probs(
            scorer.graph_model,
            graph_sample,
            device=scorer.device,
        )
        for key, prob in zip(graph_sample.edge_keys, graph_probs):
            base_score = component_lookup[key].fused
            score_lookup[key] = float(
                ((1.0 - scorer.graph_weight) * base_score) + (scorer.graph_weight * float(prob))
            )
    return oriented_tiles, score_lookup


def compute_border_priors(
    pieces: Sequence[Piece],
    score_lookup: Dict[Tuple[int, int, int, int, str], float],
    state_values: Sequence[int] = (0, 1, 2, 3),
) -> Dict[Tuple[int, int, str], float]:
    border_priors: Dict[Tuple[int, int, str], float] = {}
    piece_ids = [piece.piece_id for piece in pieces]
    for pid in piece_ids:
        for state in state_values:
            right_best = 0.0
            left_best = 0.0
            bottom_best = 0.0
            top_best = 0.0
            for other_id in piece_ids:
                if other_id == pid:
                    continue
                for other_state in state_values:
                    right_best = max(right_best, score_lookup[(pid, state, other_id, other_state, "right")])
                    left_best = max(left_best, score_lookup[(other_id, other_state, pid, state, "right")])
                    bottom_best = max(bottom_best, score_lookup[(pid, state, other_id, other_state, "down")])
                    top_best = max(top_best, score_lookup[(other_id, other_state, pid, state, "down")])
            border_priors[(pid, state, "left")] = float(np.clip(1.0 - left_best, 0.0, 1.0))
            border_priors[(pid, state, "right")] = float(np.clip(1.0 - right_best, 0.0, 1.0))
            border_priors[(pid, state, "top")] = float(np.clip(1.0 - top_best, 0.0, 1.0))
            border_priors[(pid, state, "bottom")] = float(np.clip(1.0 - bottom_best, 0.0, 1.0))
    return border_priors


def boundary_bonus_for_position(
    candidate: Tuple[int, int],
    r: int,
    c: int,
    rows: int,
    cols: int,
    border_priors: Dict[Tuple[int, int, str], float],
    border_bonus_weight: float,
) -> float:
    if border_bonus_weight <= 0.0:
        return 0.0
    pid, state = candidate
    bonus = 0.0
    if r == 0:
        bonus += border_priors[(pid, state, "top")]
    if r == rows - 1:
        bonus += border_priors[(pid, state, "bottom")]
    if c == 0:
        bonus += border_priors[(pid, state, "left")]
    if c == cols - 1:
        bonus += border_priors[(pid, state, "right")]
    return border_bonus_weight * bonus


def _candidate_local_score(
    placement_flat: Sequence[Tuple[int, int]],
    candidate: Tuple[int, int],
    next_index: int,
    rows: int,
    cols: int,
    score_lookup: Dict[Tuple[int, int, int, int, str], float],
    border_priors: Dict[Tuple[int, int, str], float],
    border_bonus_weight: float,
) -> float:
    r, c = divmod(next_index, cols)
    pid, state = candidate
    score = 0.0
    if c > 0:
        left_pid, left_state = placement_flat[next_index - 1]
        score += score_lookup[(left_pid, left_state, pid, state, "right")]
    if r > 0:
        above_pid, above_state = placement_flat[next_index - cols]
        score += score_lookup[(above_pid, above_state, pid, state, "down")]
    score += boundary_bonus_for_position(
        candidate,
        r,
        c,
        rows,
        cols,
        border_priors,
        border_bonus_weight,
    )
    return score


def reconstruct_beam_search(
    pieces: Sequence[Piece],
    rows: int,
    cols: int,
    score_lookup: Dict[Tuple[int, int, int, int, str], float],
    border_priors: Dict[Tuple[int, int, str], float],
    state_values: Sequence[int] = (0, 1, 2, 3),
    initial_pair_candidates: int = 24,
    beam_width: int = 64,
    beam_branch_factor: int = 16,
    border_bonus_weight: float = 0.35,
    progress_label: str | None = None,
) -> List[List[Tuple[int, int]]]:
    piece_ids = [piece.piece_id for piece in pieces]
    piece_set = set(piece_ids)

    if cols == 1:
        beams: List[Tuple[float, Tuple[Tuple[int, int], ...], Tuple[int, ...]]] = []
        for pid in piece_ids:
            for state in state_values:
                seed_score = 0.0
                for other_id in piece_ids:
                    if other_id == pid:
                        continue
                    seed_score = max(
                        seed_score,
                        max(score_lookup[(pid, state, other_id, other_state, "down")] for other_state in state_values),
                    )
                seed_score += boundary_bonus_for_position(
                    (pid, state),
                    0,
                    0,
                    rows,
                    cols,
                    border_priors,
                    border_bonus_weight,
                )
                unused = tuple(sorted(piece_set - {pid}))
                beams.append((seed_score, ((pid, state),), unused))
        beams.sort(key=lambda item: item[0], reverse=True)
        beams = beams[:max(1, beam_width)]
        start_index = 1
    else:
        initial_states: List[Tuple[float, Tuple[Tuple[int, int], ...], Tuple[int, ...]]] = []
        for a_id in piece_ids:
            for b_id in piece_ids:
                if a_id == b_id:
                    continue
                for a_state in state_values:
                    for b_state in state_values:
                        seed_score = score_lookup[(a_id, a_state, b_id, b_state, "right")]
                        seed_score += boundary_bonus_for_position(
                            (a_id, a_state),
                            0,
                            0,
                            rows,
                            cols,
                            border_priors,
                            border_bonus_weight,
                        )
                        seed_score += boundary_bonus_for_position(
                            (b_id, b_state),
                            0,
                            1,
                            rows,
                            cols,
                            border_priors,
                            border_bonus_weight,
                        )
                        unused = tuple(sorted(piece_set - {a_id, b_id}))
                        initial_states.append((seed_score, ((a_id, a_state), (b_id, b_state)), unused))
        initial_states.sort(key=lambda item: item[0], reverse=True)
        beams = initial_states[:max(1, min(beam_width, initial_pair_candidates))]
        start_index = 2

    for next_index in range(start_index, rows * cols):
        expanded: List[Tuple[float, Tuple[Tuple[int, int], ...], Tuple[int, ...]]] = []
        for base_score, placement_flat, unused in beams:
            local_candidates = []
            for pid in unused:
                for state in state_values:
                    local_score = _candidate_local_score(
                        placement_flat,
                        (pid, state),
                        next_index,
                        rows,
                        cols,
                        score_lookup,
                        border_priors,
                        border_bonus_weight,
                    )
                    local_candidates.append((local_score, pid, state))
            local_candidates.sort(key=lambda item: item[0], reverse=True)
            for local_score, pid, state in local_candidates[:max(1, beam_branch_factor)]:
                new_unused = tuple(x for x in unused if x != pid)
                expanded.append(
                    (
                        base_score + local_score,
                        placement_flat + ((pid, state),),
                        new_unused,
                    )
                )

        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = expanded[:max(1, beam_width)]
        if not beams:
            raise RuntimeError("Beam search failed to produce any candidate placement.")
        if progress_label:
            print_status(
                f"[{progress_label}] placed={next_index + 1}/{rows * cols} "
                f"beam_count={len(beams)} best_partial_score={beams[0][0]:.4f}"
            )

    best_flat = beams[0][1]
    return [list(best_flat[r * cols:(r + 1) * cols]) for r in range(rows)]


def arrangement_score(
    placement_ids: Sequence[Sequence[Tuple[int, int]]],
    rows: int,
    cols: int,
    score_lookup: Dict[Tuple[int, int, int, int, str], float],
    border_priors: Dict[Tuple[int, int, str], float],
    border_bonus_weight: float = 0.35,
) -> float:
    total = 0.0
    for r in range(rows):
        for c in range(cols):
            pid, state = placement_ids[r][c]
            if c < cols - 1:
                right_pid, right_state = placement_ids[r][c + 1]
                total += score_lookup[(pid, state, right_pid, right_state, "right")]
            if r < rows - 1:
                down_pid, down_state = placement_ids[r + 1][c]
                total += score_lookup[(pid, state, down_pid, down_state, "down")]
            total += boundary_bonus_for_position(
                (pid, state),
                r,
                c,
                rows,
                cols,
                border_priors,
                border_bonus_weight,
            )
    return total


def placement_tiles_from_ids(
    placement_ids: Sequence[Sequence[Tuple[int, int]]],
    oriented_tiles: Dict[Tuple[int, int], np.ndarray],
) -> List[List[np.ndarray]]:
    return [[oriented_tiles[(pid, state)] for pid, state in row] for row in placement_ids]


def optimize_placement_simulated_annealing(
    placement_ids: Sequence[Sequence[Tuple[int, int]]],
    rows: int,
    cols: int,
    score_lookup: Dict[Tuple[int, int, int, int, str], float],
    border_priors: Dict[Tuple[int, int, str], float],
    state_values: Sequence[int] = (0, 1, 2, 3),
    border_bonus_weight: float = 0.35,
    anneal_steps: int = 4000,
    start_temp: float = 0.25,
    end_temp: float = 0.01,
    progress_label: str | None = None,
) -> List[List[Tuple[int, int]]]:
    if anneal_steps <= 0:
        return [list(row) for row in placement_ids]

    positions = [(r, c) for r in range(rows) for c in range(cols)]
    current = [list(row) for row in placement_ids]
    current_score = arrangement_score(current, rows, cols, score_lookup, border_priors, border_bonus_weight)
    best = [list(row) for row in current]
    best_score = current_score
    anneal_start = time.perf_counter()
    last_progress_time = anneal_start

    for step in range(anneal_steps):
        proposal = [row[:] for row in current]
        if random.random() < 0.65:
            idx_a, idx_b = random.sample(range(len(positions)), 2)
            ra, ca = positions[idx_a]
            rb, cb = positions[idx_b]
            proposal[ra][ca], proposal[rb][cb] = proposal[rb][cb], proposal[ra][ca]
            if len(state_values) > 1 and random.random() < 0.35:
                pid_a, _ = proposal[ra][ca]
                pid_b, _ = proposal[rb][cb]
                proposal[ra][ca] = (pid_a, random.choice(state_values))
                proposal[rb][cb] = (pid_b, random.choice(state_values))
        else:
            r, c = random.choice(positions)
            pid, _ = proposal[r][c]
            proposal[r][c] = (pid, random.choice(state_values))

        proposal_score = arrangement_score(proposal, rows, cols, score_lookup, border_priors, border_bonus_weight)
        if anneal_steps == 1:
            temp = end_temp
        else:
            progress = step / max(anneal_steps - 1, 1)
            temp = start_temp * ((end_temp / start_temp) ** progress)
        accept = proposal_score >= current_score
        if not accept:
            delta = proposal_score - current_score
            accept = random.random() < math.exp(delta / max(temp, 1e-8))

        if accept:
            current = proposal
            current_score = proposal_score
            if proposal_score > best_score:
                best = [row[:] for row in proposal]
                best_score = proposal_score
        if progress_label:
            now = time.perf_counter()
            if (now - last_progress_time) >= 30.0 or step + 1 == anneal_steps:
                print_status(
                    f"[{progress_label}] anneal_step={step + 1}/{anneal_steps} "
                    f"best_score={best_score:.4f} current_score={current_score:.4f} "
                    f"elapsed={format_elapsed(now - anneal_start)}"
                )
                last_progress_time = now

    return best


def perturb_score_lookup(
    score_lookup: Dict[Tuple[int, int, int, int, str], float],
    noise_scale: float,
    rng: random.Random,
) -> Dict[Tuple[int, int, int, int, str], float]:
    if noise_scale <= 0.0:
        return dict(score_lookup)
    return {
        key: float(np.clip(score + rng.normalvariate(0.0, noise_scale), 0.0, 1.0))
        for key, score in score_lookup.items()
    }


def reconstruct_greedy(
    pieces: Sequence[Piece],
    rows: int,
    cols: int,
    scorer: FusionScorer,
    strip_size: int = 16,
    initial_pair_candidates: int = 12,
    beam_width: int = 64,
    beam_branch_factor: int = 16,
    border_bonus_weight: float = 0.35,
    fixed_orientation: bool = False,
    state_values: Sequence[int] | None = None,
    anneal_steps: int = 4000,
    global_search_restarts: int = 1,
    global_search_score_noise: float = 0.0,
) -> Tuple[List[List[Tuple[int, int]]], List[List[np.ndarray]], Dict[str, float | int]]:
    n = len(pieces)
    if n != rows * cols:
        raise ValueError("Number of pieces must equal rows * cols.")

    if fixed_orientation:
        resolved_state_values = (0,)
    elif state_values is None:
        resolved_state_values = FLIP_STATE_VALUES
    else:
        resolved_state_values = tuple(state_values)
        if not resolved_state_values:
            raise ValueError("state_values must not be empty.")
    resolved_restarts = max(int(global_search_restarts), 1)
    print_status(
        f"[Global Search] scoring pieces={n} states={len(resolved_state_values)} "
        f"restarts={resolved_restarts}"
    )
    oriented_tiles, score_lookup = precompute_score_lookup(
        pieces,
        scorer,
        strip_size=strip_size,
        state_values=resolved_state_values,
        progress_label=f"Global Search Score {rows}x{cols}",
    )
    print_status("[Global Search] computing border priors")
    border_priors = compute_border_priors(pieces, score_lookup, state_values=resolved_state_values)
    search_candidates = []
    print_status("[Global Search] building beam candidate 1")
    base_candidate = reconstruct_beam_search(
        pieces,
        rows,
        cols,
        score_lookup,
        border_priors,
        state_values=resolved_state_values,
        initial_pair_candidates=initial_pair_candidates,
        beam_width=beam_width,
        beam_branch_factor=beam_branch_factor,
        border_bonus_weight=border_bonus_weight,
        progress_label=f"Beam Search 1/{resolved_restarts}",
    )
    search_candidates.append(base_candidate)

    if resolved_restarts > 1:
        rng = random.Random(SEED + (rows * 97) + (cols * 193) + (len(pieces) * 17))
        for restart_index in range(2, resolved_restarts + 1):
            print_status(f"[Global Search] building beam candidate {restart_index}")
            perturbed_lookup = perturb_score_lookup(score_lookup, global_search_score_noise, rng)
            perturbed_border_priors = compute_border_priors(
                pieces,
                perturbed_lookup,
                state_values=resolved_state_values,
            )
            search_candidates.append(
                reconstruct_beam_search(
                    pieces,
                    rows,
                    cols,
                    perturbed_lookup,
                    perturbed_border_priors,
                    state_values=resolved_state_values,
                    initial_pair_candidates=initial_pair_candidates,
                    beam_width=beam_width,
                    beam_branch_factor=beam_branch_factor,
                    border_bonus_weight=border_bonus_weight,
                    progress_label=f"Beam Search {restart_index}/{resolved_restarts}",
                )
            )

    best_ids = None
    best_score = float("-inf")
    for candidate_index, candidate_ids in enumerate(search_candidates, start=1):
        print_status(f"[Global Search] optimizing candidate {candidate_index}/{len(search_candidates)}")
        optimized_ids = optimize_placement_simulated_annealing(
            candidate_ids,
            rows,
            cols,
            score_lookup,
            border_priors,
            state_values=resolved_state_values,
            border_bonus_weight=border_bonus_weight,
            anneal_steps=anneal_steps,
            progress_label=f"Anneal {candidate_index}/{len(search_candidates)}",
        )
        optimized_score = arrangement_score(
            optimized_ids,
            rows,
            cols,
            score_lookup,
            border_priors,
            border_bonus_weight=border_bonus_weight,
        )
        print_status(
            f"[Global Search] candidate {candidate_index}/{len(search_candidates)} "
            f"score={optimized_score:.4f}"
        )
        if best_ids is None or optimized_score > best_score:
            best_ids = optimized_ids
            best_score = optimized_score

    if best_ids is None:
        raise RuntimeError("Global reconstruction search did not produce any candidate arrangement.")

    optimized_tiles = placement_tiles_from_ids(best_ids, oriented_tiles)
    return best_ids, optimized_tiles, {
        "search_candidates_evaluated": len(search_candidates),
        "best_arrangement_score": float(best_score),
    }

def position_accuracy(placement_ids: List[List[Tuple[int, int]]], cols: int) -> float:
    total = 0
    correct = 0
    for r in range(len(placement_ids)):
        for c in range(len(placement_ids[0])):
            pid, _ = placement_ids[r][c]
            total += 1
            gt_r, gt_c = index_to_coord(pid, cols)
            if gt_r == r and gt_c == c:
                correct += 1
    return correct / max(total, 1)


def adjacency_accuracy(placement_ids: List[List[Tuple[int, int]]], rows: int, cols: int) -> float:
    total = 0
    correct = 0
    for r in range(rows):
        for c in range(cols):
            pid, _ = placement_ids[r][c]
            if c < cols - 1:
                total += 1
                right_pid, _ = placement_ids[r][c + 1]
                if are_neighbors(pid, right_pid, rows, cols, "right"):
                    correct += 1
            if r < rows - 1:
                total += 1
                down_pid, _ = placement_ids[r + 1][c]
                if are_neighbors(pid, down_pid, rows, cols, "down"):
                    correct += 1
    return correct / max(total, 1)


def orientation_accuracy(placement_ids: List[List[Tuple[int, int]]], pieces: Sequence[Piece]) -> float:
    piece_lookup = {piece.piece_id: piece for piece in pieces}
    predicted = {
        pid: state
        for row in placement_ids
        for pid, state in row
    }
    total = len(piece_lookup)
    correct = sum(
        1
        for pid, piece in piece_lookup.items()
        if predicted.get(pid, -1) == piece.scramble_state
    )
    return correct / max(total, 1)


def joint_position_orientation_accuracy(
    placement_ids: List[List[Tuple[int, int]]],
    pieces: Sequence[Piece],
    cols: int,
) -> float:
    piece_lookup = {piece.piece_id: piece for piece in pieces}
    total = 0
    correct = 0
    for r in range(len(placement_ids)):
        for c in range(len(placement_ids[0])):
            pid, state = placement_ids[r][c]
            total += 1
            piece = piece_lookup[pid]
            gt_r, gt_c = piece.gt_position
            if gt_r == r and gt_c == c and state == piece.scramble_state:
                correct += 1
    return correct / max(total, 1)


def image_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    return float(ssim(gray_a, gray_b, data_range=255))


def exact_image_match(img_a: np.ndarray, img_b: np.ndarray) -> bool:
    return img_a.shape == img_b.shape and bool(np.array_equal(img_a, img_b))


def pixel_match_ratio(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if img_a.shape != img_b.shape:
        return 0.0
    per_pixel = np.all(img_a == img_b, axis=-1)
    return float(per_pixel.mean())


def merge_or_visualize_tiles(tiles_grid: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    try:
        return merge_tiles(tiles_grid)
    except ValueError:
        return build_visual_grid(tiles_grid)


def evaluate_reconstruction_result(
    placement_ids: List[List[Tuple[int, int]]],
    placement_tiles: List[List[np.ndarray]],
    pieces: Sequence[Piece],
    rows: int,
    cols: int,
    reference_img: np.ndarray,
    reference_visual_img: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float | bool]]:
    reconstructed_img = merge_or_visualize_tiles(placement_tiles)
    metrics = {
        "position_accuracy": position_accuracy(placement_ids, cols),
        "adjacency_accuracy": adjacency_accuracy(placement_ids, rows, cols),
        "orientation_accuracy": orientation_accuracy(placement_ids, pieces),
        "joint_position_orientation_accuracy": joint_position_orientation_accuracy(placement_ids, pieces, cols),
        "ssim": image_ssim(reference_visual_img, reconstructed_img),
        "pixel_match_ratio": pixel_match_ratio(reference_img, reconstructed_img),
        "exact_match": exact_image_match(reference_img, reconstructed_img),
    }
    return reconstructed_img, metrics


def print_reconstruction_metrics(metrics: Dict[str, float | bool]) -> None:
    print("[Reconstruction Result]")
    labels = (
        ("Position accuracy", "position_accuracy"),
        ("Adjacency accuracy", "adjacency_accuracy"),
        ("Orientation accuracy", "orientation_accuracy"),
        ("Joint pos+orientation acc", "joint_position_orientation_accuracy"),
        ("SSIM", "ssim"),
        ("Pixel match ratio", "pixel_match_ratio"),
    )
    for label, key in labels:
        print(f"{label:<28}: {float(metrics[key]):.4f}")
    print(f"{'Exact match':<28}: {metrics['exact_match']}")


def save_experiment_outputs(
    out_dir: Path,
    reference_img: np.ndarray,
    scrambled_img: np.ndarray,
    reconstructed_img: np.ndarray,
    final_metrics: Dict[str, object],
    rf: RandomForestClassifier | None,
    nn_model: PairNet | None,
    graph_model: GraphEdgeReranker | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_outputs = {
        "original.png": reference_img,
        "scrambled.png": scrambled_img,
        "reference_reassembly.png": reference_img,
        "reconstructed.png": reconstructed_img,
    }
    for filename, image in image_outputs.items():
        save_rgb_image(out_dir / filename, image)
    if rf is not None:
        with (out_dir / "random_forest.pkl").open("wb") as handle:
            pickle.dump(rf, handle)
    if nn_model is not None:
        torch.save(
            {
                "state_dict": nn_model.state_dict(),
                "input_mode": nn_model.input_mode,
            },
            out_dir / "pairnet.pt",
        )
    if graph_model is not None:
        torch.save(
            {
                "state_dict": graph_model.state_dict(),
                "node_input_dim": graph_model.node_input_dim,
                "edge_input_dim": graph_model.edge_input_dim,
                "hidden_dim": graph_model.hidden_dim,
                "num_layers": graph_model.num_layers,
                "model_type": graph_model.model_type,
            },
            out_dir / "graph_reranker.pt",
        )
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")


def build_reference_tiles(
    test_image_path: str | Path,
    rows: int,
    cols: int,
    prefer_existing_slices: bool = True,
    slice_root: str | Path | None = None,
) -> Tuple[List[np.ndarray], List[Piece] | None, str]:
    test_image_path = Path(test_image_path)
    slice_folder = find_slice_folder_for_image(test_image_path, rows, cols, slice_root=slice_root)
    if prefer_existing_slices and slice_folder is not None:
        if read_slice_metadata(slice_folder) is not None:
            tiles, pieces = load_randomized_tiles_and_pieces(slice_folder, rows, cols)
            return tiles, pieces, f"randomized_slice_folder:{slice_folder}"
        return load_tiles_from_folder(slice_folder), None, f"slice_folder:{slice_folder}"

    test_img = read_image(test_image_path)
    return split_into_tiles(test_img, rows, cols), None, "resized_full_image"


EXTRA_RUN_ARG_KEYS = {"model_variant", "model_label", "base_edge_variant"}
MODEL_VARIANT_ORDER = ("random_forest", "pairnet", "fusion", "graphsage", "gat")
MODEL_INTERFACE_SCRIPT_NAMES = {
    "random_forest": "pathology_reconstruction_random_forest.py",
    "pairnet": "pathology_reconstruction_pairnet.py",
    "fusion": "pathology_reconstruction_fusion.py",
    "graphsage": "pathology_reconstruction_graphsage.py",
    "gat": "pathology_reconstruction_gat.py",
}
# Grid presets used by the CLI and by model-interface scripts.
# The project currently contains slice folders for 3x6, 4x5, 5x4, 5x7, and 10x8.
# 5x5 is kept as an optional legacy/previous-paper setting.
GRID_CHOICE_PRESETS = {
    "1": (3, 6),
    "2": (4, 5),
    "3": (5, 4),
    "4": (5, 7),
    "5": (10, 8),
    "6": (5, 5),
}


@dataclass(frozen=True)
class CliGridFlags:
    rows_specified: bool
    cols_specified: bool
    grid_choice_specified: bool
    help_requested: bool


def list_model_variants() -> Tuple[str, ...]:
    return MODEL_VARIANT_ORDER


def get_model_interface_script_name(variant_name: str) -> str:
    if variant_name not in MODEL_INTERFACE_SCRIPT_NAMES:
        raise ValueError(f"Unsupported model variant: {variant_name}")
    return MODEL_INTERFACE_SCRIPT_NAMES[variant_name]


def grid_choice_to_shape(choice: str) -> Tuple[int, int]:
    resolved_choice = str(choice).strip()
    if resolved_choice not in GRID_CHOICE_PRESETS:
        raise ValueError(f"Unsupported grid choice: {choice}")
    return GRID_CHOICE_PRESETS[resolved_choice]


def has_cli_flag(argv: Sequence[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def normalize_cli_argv(argv: Sequence[str] | None = None) -> List[str]:
    return list(argv) if argv is not None else []


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    normalized_argv = normalize_cli_argv(argv)
    args = build_argparser().parse_args(normalized_argv)
    # Apply --grid_choice even when the caller uses run_model_variant_cli() or
    # build_run_args(), not only the interactive interface runner. This prevents
    # rows/cols from silently staying at the 4x5 defaults when --grid_choice is
    # supplied from notebooks or wrapper scripts.
    cli_grid_flags = inspect_cli_grid_flags(normalized_argv)
    apply_grid_choice_to_args(args, cli_grid_flags)
    return args


def inspect_cli_grid_flags(argv: Sequence[str]) -> CliGridFlags:
    return CliGridFlags(
        rows_specified=has_cli_flag(argv, "--rows"),
        cols_specified=has_cli_flag(argv, "--cols"),
        grid_choice_specified=has_cli_flag(argv, "--grid_choice"),
        help_requested=has_cli_flag(argv, "--help") or has_cli_flag(argv, "-h"),
    )


def prompt_grid_choice(variant_name: str) -> str:
    prompt_lines = (
        "",
        f"[{variant_name}] Select grid preset:",
        "  1 -> 3x6",
        "  2 -> 4x5",
        "  3 -> 5x4",
        "  4 -> 5x7",
        "  5 -> 10x8",
        "  6 -> 5x5",
        "Enter 1, 2, 3, 4, 5, or 6 (default: 2): ",
    )
    if not sys.stdin or not sys.stdin.isatty():
        print(f"[{variant_name}] No interactive input detected; defaulting to grid preset 2 (4x5).")
        return "2"

    while True:
        try:
            choice = input("\n".join(prompt_lines)).strip()
        except EOFError:
            print(f"[{variant_name}] Input stream closed; defaulting to grid preset 2 (4x5).")
            return "2"
        if choice == "":
            return "2"
        if choice in GRID_CHOICE_PRESETS:
            return choice
        print("Invalid selection. Please enter one of: " + ", ".join(GRID_CHOICE_PRESETS.keys()) + ".")


def maybe_append_prompted_grid_choice(
    argv: Sequence[str],
    variant_name: str,
    cli_grid_flags: CliGridFlags,
) -> List[str]:
    normalized_argv = list(argv)
    if (
        cli_grid_flags.help_requested
        or cli_grid_flags.rows_specified
        or cli_grid_flags.cols_specified
        or cli_grid_flags.grid_choice_specified
    ):
        return normalized_argv

    selected_choice = prompt_grid_choice(variant_name)
    selected_rows, selected_cols = grid_choice_to_shape(selected_choice)
    normalized_argv.extend(
        [
            "--grid_choice",
            selected_choice,
            "--rows",
            str(selected_rows),
            "--cols",
            str(selected_cols),
        ]
    )
    return normalized_argv


def apply_grid_choice_to_args(args: argparse.Namespace, cli_grid_flags: CliGridFlags) -> None:
    if args.grid_choice and not (cli_grid_flags.rows_specified and cli_grid_flags.cols_specified):
        args.rows, args.cols = grid_choice_to_shape(args.grid_choice)


def resolve_cli_output_dir(
    args: argparse.Namespace,
    argv: Sequence[str],
    *,
    variant_name: str | None = None,
    include_grid: bool = False,
) -> str:
    if has_cli_flag(argv, "--output_dir"):
        return args.output_dir

    resolved_output_dir = Path(args.output_dir)
    if include_grid:
        resolved_output_dir /= f"{args.rows}x{args.cols}"
    if variant_name is not None:
        resolved_output_dir /= variant_name
    return str(resolved_output_dir)


def resolve_model_variant_spec(base_args: argparse.Namespace, variant_name: str) -> Dict[str, object]:
    if variant_name == "random_forest":
        return {
            "label": "Random Forest",
            "overrides": {
                "rf_weight": 1.0,
                "graph_weight": 0.0,
                "graph_model_type": None,
            },
            "extra_attrs": {
                "model_variant": variant_name,
                "model_label": "Random Forest",
                "base_edge_variant": None,
            },
        }
    if variant_name == "pairnet":
        return {
            "label": "PairNet",
            "overrides": {
                "rf_weight": 0.0,
                "graph_weight": 0.0,
                "graph_model_type": None,
            },
            "extra_attrs": {
                "model_variant": variant_name,
                "model_label": "PairNet",
                "base_edge_variant": None,
            },
        }
    if variant_name == "fusion":
        return {
            "label": "RF + PairNet Fusion",
            "overrides": {
                "rf_weight": getattr(base_args, "rf_weight", 0.5),
                "graph_weight": 0.0,
                "graph_model_type": None,
            },
            "extra_attrs": {
                "model_variant": variant_name,
                "model_label": "RF + PairNet Fusion",
                "base_edge_variant": None,
            },
        }
    if variant_name == "graphsage":
        return {
            "label": "GraphSAGE Reranker",
            "overrides": {
                "rf_weight": getattr(base_args, "rf_weight", 0.5),
                "graph_weight": max(getattr(base_args, "graph_weight", 0.5), 0.5),
                "graph_model_type": "graphsage",
            },
            "extra_attrs": {
                "model_variant": variant_name,
                "model_label": "GraphSAGE Reranker",
                "base_edge_variant": "fusion",
            },
        }
    if variant_name == "gat":
        return {
            "label": "GAT Reranker",
            "overrides": {
                "rf_weight": getattr(base_args, "rf_weight", 0.5),
                "graph_weight": max(getattr(base_args, "graph_weight", 0.5), 0.5),
                "graph_model_type": "gat",
            },
            "extra_attrs": {
                "model_variant": variant_name,
                "model_label": "GAT Reranker",
                "base_edge_variant": "fusion",
            },
        }
    raise ValueError(f"Unsupported model variant: {variant_name}")


def build_run_args(**overrides) -> argparse.Namespace:
    args = parse_cli_args([])
    valid_keys = set(vars(args))
    unknown_keys = sorted(set(overrides) - valid_keys - EXTRA_RUN_ARG_KEYS)
    if unknown_keys:
        raise ValueError(f"Unsupported run-arg overrides: {', '.join(unknown_keys)}")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def clone_run_args(base_args: argparse.Namespace, **overrides) -> argparse.Namespace:
    if not isinstance(base_args, argparse.Namespace):
        raise TypeError("base_args must be an argparse.Namespace")
    return build_run_args(**{**vars(base_args), **overrides})


def summarize_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    keys = [
        "model_variant",
        "model_label",
        "base_edge_variant",
        "feature_mode",
        "cnn_input_mode",
        "stain_normalization",
        "rf_weight",
        "graph_weight",
        "graph_model_type",
        "global_search_restarts",
        "global_search_score_noise",
        "training_data_source",
        "validation_split_strategy",
        "position_accuracy",
        "adjacency_accuracy",
        "orientation_accuracy",
        "joint_position_orientation_accuracy",
        "ssim",
        "pixel_match_ratio",
        "model_exact_match",
        "reference_exact_match",
        "graph_validation_auc",
        "graph_validation_ap",
        "graph_validation_acc",
        "search_candidates_evaluated",
        "best_arrangement_score",
        "output_dir",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def flatten_metrics_for_table(metrics: Dict[str, object]) -> Dict[str, object]:
    test_image = metrics.get("test_image")
    test_image_path = Path(test_image) if isinstance(test_image, str) and test_image else None
    return {
        "test_image_name": None if test_image_path is None else test_image_path.name,
        "test_image": test_image,
        "model_variant": metrics.get("model_variant"),
        "model_label": metrics.get("model_label"),
        "rows": metrics.get("rows"),
        "cols": metrics.get("cols"),
        "feature_mode": metrics.get("feature_mode"),
        "cnn_input_mode": metrics.get("cnn_input_mode"),
        "stain_normalization": metrics.get("stain_normalization"),
        "rf_weight": metrics.get("rf_weight"),
        "graph_weight": metrics.get("graph_weight"),
        "graph_model_type": metrics.get("graph_model_type"),
        "training_data_source": metrics.get("training_data_source"),
        "validation_split_strategy": metrics.get("validation_split_strategy"),
        "epochs": metrics.get("epochs"),
        "position_accuracy": metrics.get("position_accuracy"),
        "adjacency_accuracy": metrics.get("adjacency_accuracy"),
        "orientation_accuracy": metrics.get("orientation_accuracy"),
        "joint_position_orientation_accuracy": metrics.get("joint_position_orientation_accuracy"),
        "ssim": metrics.get("ssim"),
        "pixel_match_ratio": metrics.get("pixel_match_ratio"),
        "model_exact_match": metrics.get("model_exact_match"),
        "graph_validation_auc": metrics.get("graph_validation_auc"),
        "graph_validation_ap": metrics.get("graph_validation_ap"),
        "graph_validation_acc": metrics.get("graph_validation_acc"),
        "global_search_restarts": metrics.get("global_search_restarts"),
        "global_search_score_noise": metrics.get("global_search_score_noise"),
        "search_candidates_evaluated": metrics.get("search_candidates_evaluated"),
        "best_arrangement_score": metrics.get("best_arrangement_score"),
        "total_seconds": None if not isinstance(metrics.get("timings"), dict) else metrics["timings"].get("total_seconds"),
        "output_dir": metrics.get("output_dir"),
    }


def write_metrics_rows_csv(output_path: str | Path, rows: Sequence[Dict[str, object]]) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty metrics table.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    serialized[key] = json.dumps(value, ensure_ascii=False)
                else:
                    serialized[key] = value
            writer.writerow(serialized)
    return output_path


def aggregate_table_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot aggregate zero rows.")

    metric_keys = (
        "position_accuracy",
        "adjacency_accuracy",
        "orientation_accuracy",
        "joint_position_orientation_accuracy",
        "ssim",
        "pixel_match_ratio",
        "graph_validation_auc",
        "graph_validation_ap",
        "graph_validation_acc",
        "best_arrangement_score",
        "total_seconds",
    )
    aggregate = {
        "test_image_name": "__mean__",
        "test_image": "__mean__",
        "image_count": len(rows),
        "exact_match_rate": float(np.mean([float(bool(row.get("model_exact_match"))) for row in rows])),
    }
    for key in metric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        aggregate[key] = None if not values else float(np.mean(values))
    return aggregate


def save_single_run_summary_table(out_dir: Path, metrics: Dict[str, object]) -> Path:
    return write_metrics_rows_csv(out_dir / "metrics_summary.csv", [flatten_metrics_for_table(metrics)])


def build_model_variant_args(
    base_args: argparse.Namespace,
    variant_name: str,
    output_dir: str | Path | None = None,
) -> argparse.Namespace:
    spec = resolve_model_variant_spec(base_args, variant_name)
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(base_args.output_dir) / variant_name
    variant_args = clone_run_args(
        base_args,
        output_dir=str(resolved_output_dir),
        **spec["overrides"],
        **spec["extra_attrs"],
    )
    return variant_args


def run_model_variant(
    base_args: argparse.Namespace,
    variant_name: str,
    output_dir: str | Path | None = None,
) -> Dict[str, object]:
    variant_args = build_model_variant_args(base_args, variant_name, output_dir=output_dir)
    print(f"\n[Model Variant] {variant_args.model_label} ({variant_args.model_variant})")
    return run_from_args(variant_args)


def run_model_suite(
    base_args: argparse.Namespace,
    variant_names: Sequence[str] | None = None,
    output_root: str | Path | None = None,
) -> Dict[str, Dict[str, object]]:
    resolved_variants = tuple(variant_names) if variant_names is not None else MODEL_VARIANT_ORDER
    results: Dict[str, Dict[str, object]] = {}
    for variant_name in resolved_variants:
        variant_output_dir = None if output_root is None else Path(output_root) / variant_name
        results[variant_name] = run_model_variant(
            base_args,
            variant_name,
            output_dir=variant_output_dir,
        )
    if results:
        summary_root = Path(output_root) if output_root is not None else Path(base_args.output_dir)
        write_metrics_rows_csv(
            summary_root / "model_variant_summary.csv",
            [flatten_metrics_for_table(metrics) for metrics in results.values()],
        )
    return results


def run_model_variant_cli(variant_name: str, argv: Sequence[str] | None = None) -> Dict[str, object]:
    normalized_argv = normalize_cli_argv(argv)
    args = parse_cli_args(normalized_argv)
    resolved_output_dir = resolve_cli_output_dir(args, normalized_argv, variant_name=variant_name)
    return run_model_variant(args, variant_name, output_dir=resolved_output_dir)


def run_model_interface_cli(variant_name: str, argv: Sequence[str] | None = None) -> Dict[str, object]:
    normalized_argv = normalize_cli_argv(argv)
    cli_grid_flags = inspect_cli_grid_flags(normalized_argv)
    resolved_argv = maybe_append_prompted_grid_choice(normalized_argv, variant_name, cli_grid_flags)
    args = parse_cli_args(resolved_argv)
    apply_grid_choice_to_args(args, cli_grid_flags)
    resolved_output_dir = resolve_cli_output_dir(
        args,
        resolved_argv,
        variant_name=variant_name,
        include_grid=True,
    )
    return run_model_variant(args, variant_name, output_dir=resolved_output_dir)


def run_ablation_suite(base_args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    variant_specs = {
        "color_rf": dict(feature_mode="color", rf_weight=1.0, graph_weight=0.0),
        "texture_rf": dict(feature_mode="texture", rf_weight=1.0, graph_weight=0.0),
        "stain_rf": dict(feature_mode="stain", rf_weight=1.0, graph_weight=0.0),
        "cnn_only": dict(feature_mode="fusion", rf_weight=0.0, graph_weight=0.0),
        "fusion_base": dict(feature_mode="fusion", rf_weight=base_args.rf_weight, graph_weight=0.0),
        "fusion_graphsage": dict(
            feature_mode="fusion",
            rf_weight=base_args.rf_weight,
            graph_weight=max(base_args.graph_weight, 0.5),
            graph_model_type="graphsage",
        ),
        "fusion_gat": dict(
            feature_mode="fusion",
            rf_weight=base_args.rf_weight,
            graph_weight=max(base_args.graph_weight, 0.5),
            graph_model_type="gat",
        ),
    }
    results: Dict[str, Dict[str, object]] = {}
    summary_rows: List[Dict[str, object]] = []
    for name, overrides in variant_specs.items():
        variant_dir = Path(base_args.output_dir) / name
        variant_args = clone_run_args(base_args, output_dir=str(variant_dir), **overrides)
        print(f"\n[Ablation] {name}")
        metrics = run_experiment(variant_args)
        results[name] = summarize_metrics(metrics)
        summary_rows.append(flatten_metrics_for_table(metrics))
    if summary_rows:
        write_metrics_rows_csv(Path(base_args.output_dir) / "ablation_summary.csv", summary_rows)
    return results


def run_experiment(args):
    overall_start = time.perf_counter()
    if args.cpu:
        raise RuntimeError("CPU mode has been disabled. This program now requires CUDA GPU training.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required, but CUDA is not available in the current PyTorch environment."
        )

    device = "cuda"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    train_images, test_image_path = resolve_runtime_inputs(args.train_dir, args.test_image)
    print(f"Training images: {len(train_images)}")
    print(f"Test image    : {test_image_path}")
    stain_normalizer = build_stain_normalizer(
        train_images if train_images else [test_image_path],
        method=args.stain_norm,
        sample_image_limit=args.stain_norm_fit_images,
        sample_pixels_per_image=args.stain_norm_fit_pixels,
    )
    if stain_normalizer is None:
        print("Stain normalization: disabled")
    else:
        print(
            "Stain normalization: "
            f"{stain_normalizer.method} "
            f"(fit_images={stain_normalizer.reference_image_count}, "
            f"fit_pixels={stain_normalizer.reference_pixel_count})"
        )
    print(
        "Global search   : "
        f"restarts={max(int(args.global_search_restarts), 1)} "
        f"score_noise={float(args.global_search_score_noise):.4f}"
    )

    train_pairs, val_pairs, split_metadata = build_pair_sets_for_experiment(
        train_images,
        rows=args.rows,
        cols=args.cols,
        negatives_per_positive=args.negatives_per_positive,
        use_existing_slices=args.use_existing_slices,
        slice_root=args.slice_root,
        val_ratio=args.val_ratio,
    )
    for label, value in (
        ("Training data", split_metadata["data_source"]),
        ("Split strategy", split_metadata["split_strategy"]),
        ("Train split imgs", len(split_metadata["train_split_images"])),
        ("Val split imgs", len(split_metadata["val_split_images"])),
    ):
        print(f"{label:<16}: {value}")
    print(f"Pair samples    : train={len(train_pairs)}  val={len(val_pairs)}")

    reference_tiles, reference_pieces, reference_source = build_reference_tiles(
        test_image_path,
        args.rows,
        args.cols,
        prefer_existing_slices=args.use_existing_slices,
        slice_root=args.slice_root,
    )
    print(f"Reconstruction reference: {reference_source}")
    reference_tile_grid = gridify_tiles(reference_tiles, args.rows, args.cols)
    reference_img = merge_tiles(reference_tile_grid)
    reference_visual_img = build_visual_grid(reference_tile_grid)
    reference_exact_match = exact_image_match(read_image(test_image_path), reference_img)
    print(f"Reference equals original file pixels: {reference_exact_match}")
    using_randomized_slice_pieces = reference_pieces is not None
    if using_randomized_slice_pieces:
        pieces = reference_pieces
        orientation_state_values = ALL_ORIENTATION_STATE_VALUES
    else:
        random.seed(SEED)
        np.random.seed(SEED)
        pieces, _ = scramble_tile_list(
            reference_tiles,
            args.rows,
            args.cols,
            allow_flips=not args.no_flips,
        )
        orientation_state_values = FLIP_STATE_VALUES
    fixed_orientation = args.no_flips and not using_randomized_slice_pieces
    graph_state_values = (0,) if fixed_orientation else orientation_state_values
    graph_use_existing_slices = split_metadata["data_source"] == "existing_slice_folders"
    scrambled_img = build_visual_grid(gridify_tiles([p.image for p in pieces], args.rows, args.cols))

    epoch_step = max(1, args.epoch_step)
    max_epochs = max(args.epochs, args.max_epochs)
    history = []
    best_metrics = None
    best_reconstructed_img = None
    best_key = (-1.0, -1.0, -1.0, -1.0, -1.0)
    total_nn_seconds = 0.0
    total_graph_seconds = 0.0
    total_recon_seconds = 0.0
    rf_seconds = 0.0
    rf_val_accuracy = None
    rf = None
    nn_model = None
    best_nn_model = None
    best_graph_model = None
    best_graph_validation = None
    trained_pairnet_epochs = 0

    if args.rf_weight > 0.0:
        rf_start = time.perf_counter()
        x_train, y_train = prepare_ml_arrays(
            train_pairs,
            args.strip_size,
            feature_mode=args.feature_mode,
            stain_normalizer=stain_normalizer,
        )
        x_val, y_val = prepare_ml_arrays(
            val_pairs,
            args.strip_size,
            feature_mode=args.feature_mode,
            stain_normalizer=stain_normalizer,
        )
        rf = train_random_forest(x_train, y_train, n_jobs=args.rf_jobs)
        val_pred = rf.predict(x_val)
        rf_val_accuracy = float(accuracy_score(y_val, val_pred))
        print("\n[RandomForest Validation]")
        print("Accuracy:", rf_val_accuracy)
        print(classification_report(y_val, val_pred, digits=4, zero_division=0))
        rf_seconds = time.perf_counter() - rf_start

    attempt_epoch_values = list(range(args.epochs, max_epochs + 1, epoch_step))
    if args.rf_weight >= 1.0 and args.graph_weight <= 0.0:
        attempt_epoch_values = [args.epochs]

    for total_epochs in attempt_epoch_values:
        print(f"\n[Training Attempt] total_epochs={total_epochs}")
        nn_seconds = 0.0
        graph_seconds = 0.0
        graph_model = None
        graph_validation = None

        if args.rf_weight < 1.0:
            nn_start = time.perf_counter()
            nn_epoch_delta = total_epochs if nn_model is None else max(total_epochs - trained_pairnet_epochs, 0)
            if nn_epoch_delta <= 0:
                nn_epoch_delta = epoch_step
            nn_model = train_pairnet(
                train_pairs,
                val_pairs,
                strip_size=args.strip_size,
                epochs=nn_epoch_delta,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                input_mode=args.cnn_input_mode,
                model=nn_model,
                stain_normalizer=stain_normalizer,
            )
            trained_pairnet_epochs = total_epochs
            nn_seconds = time.perf_counter() - nn_start
            total_nn_seconds += nn_seconds
        else:
            nn_model = None

        base_scorer = FusionScorer(
            rf,
            nn_model,
            device=device,
            alpha=args.rf_weight,
            feature_mode=args.feature_mode,
            stain_normalizer=stain_normalizer,
        )
        if args.graph_weight > 0.0:
            graph_start = time.perf_counter()
            train_graph_images = limit_image_paths(
                split_metadata["train_split_images"],
                getattr(args, "graph_train_image_limit", 0),
            )
            val_graph_images = limit_image_paths(
                split_metadata["val_split_images"],
                getattr(args, "graph_val_image_limit", 0),
            )
            train_graphs = build_graph_samples_from_image_paths(
                train_graph_images,
                rows=args.rows,
                cols=args.cols,
                scorer=base_scorer,
                use_existing_slices=graph_use_existing_slices,
                slice_root=args.slice_root,
                strip_size=args.strip_size,
                state_values=graph_state_values,
                candidate_k=args.graph_candidate_k,
            )
            val_graphs = build_graph_samples_from_image_paths(
                val_graph_images,
                rows=args.rows,
                cols=args.cols,
                scorer=base_scorer,
                use_existing_slices=graph_use_existing_slices,
                slice_root=args.slice_root,
                strip_size=args.strip_size,
                state_values=graph_state_values,
                candidate_k=args.graph_candidate_k,
            )
            if not val_graphs and len(train_graphs) > 1:
                split_idx = max(1, min(len(train_graphs) - 1, int(round(len(train_graphs) * 0.8))))
                val_graphs = train_graphs[split_idx:]
                train_graphs = train_graphs[:split_idx]
            print(
                f"[Graph Reranker] train_graphs={len(train_graphs)} "
                f"val_graphs={len(val_graphs)} states={len(graph_state_values)} "
                f"candidate_k={args.graph_candidate_k}"
            )
            graph_model, graph_validation = train_graph_reranker(
                train_graphs,
                val_graphs,
                device=device,
                hidden_dim=args.graph_hidden_dim,
                num_layers=args.graph_layers,
                epochs=args.graph_epochs,
                lr=args.graph_lr,
                hard_negative_ratio=args.graph_hard_negative_ratio,
                model_type=args.graph_model_type,
            )
            graph_seconds = time.perf_counter() - graph_start
            total_graph_seconds += graph_seconds

        scorer = FusionScorer(
            rf,
            nn_model,
            device=device,
            alpha=args.rf_weight,
            feature_mode=args.feature_mode,
            stain_normalizer=stain_normalizer,
            graph_model=graph_model,
            graph_weight=args.graph_weight,
            graph_candidate_k=args.graph_candidate_k,
        )

        recon_start = time.perf_counter()
        placement_ids, placement_tiles, search_metadata = reconstruct_greedy(
            pieces,
            args.rows,
            args.cols,
            scorer,
            strip_size=args.strip_size,
            initial_pair_candidates=args.initial_pair_candidates,
            beam_width=args.beam_width,
            beam_branch_factor=args.beam_branch_factor,
            border_bonus_weight=args.border_bonus_weight,
            fixed_orientation=fixed_orientation,
            state_values=orientation_state_values,
            anneal_steps=args.anneal_steps,
            global_search_restarts=args.global_search_restarts,
            global_search_score_noise=args.global_search_score_noise,
        )
        reconstructed_img, reconstruction_metrics = evaluate_reconstruction_result(
            placement_ids,
            placement_tiles,
            pieces,
            rows=args.rows,
            cols=args.cols,
            reference_img=reference_img,
            reference_visual_img=reference_visual_img,
        )
        recon_seconds = time.perf_counter() - recon_start
        total_recon_seconds += recon_seconds

        metrics = {
            "epochs": total_epochs,
            "position_accuracy": reconstruction_metrics["position_accuracy"],
            "adjacency_accuracy": reconstruction_metrics["adjacency_accuracy"],
            "orientation_accuracy": reconstruction_metrics["orientation_accuracy"],
            "joint_position_orientation_accuracy": reconstruction_metrics["joint_position_orientation_accuracy"],
            "ssim": reconstruction_metrics["ssim"],
            "pixel_match_ratio": reconstruction_metrics["pixel_match_ratio"],
            "exact_match": reconstruction_metrics["exact_match"],
            "pairnet_seconds": nn_seconds,
            "graph_reranker_seconds": graph_seconds,
            "reconstruction_seconds": recon_seconds,
            "rf_validation_accuracy": rf_val_accuracy,
            "graph_validation_auc": None if graph_validation is None else graph_validation["auc"],
            "graph_validation_ap": None if graph_validation is None else graph_validation["ap"],
            "graph_validation_acc": None if graph_validation is None else graph_validation["acc"],
            "search_candidates_evaluated": search_metadata["search_candidates_evaluated"],
            "best_arrangement_score": search_metadata["best_arrangement_score"],
        }
        history.append(metrics)
        print_reconstruction_metrics(metrics)

        result_key = (
            1.0 if metrics["exact_match"] else 0.0,
            metrics["joint_position_orientation_accuracy"],
            metrics["pixel_match_ratio"],
            metrics["position_accuracy"],
            metrics["ssim"],
        )
        if result_key > best_key:
            best_key = result_key
            best_metrics = metrics
            best_reconstructed_img = reconstructed_img
            best_nn_model = nn_model
            best_graph_model = graph_model
            best_graph_validation = graph_validation

        if metrics["exact_match"]:
            break

    if best_metrics is None or best_reconstructed_img is None:
        raise RuntimeError("No reconstruction result was produced.")

    exact_match_reached = bool(best_metrics["exact_match"])
    out_dir = Path(args.output_dir)
    model_variant = getattr(args, "model_variant", "custom")
    model_label = getattr(args, "model_label", "Custom Experiment")
    base_edge_variant = getattr(args, "base_edge_variant", None)
    artifacts = {
        "random_forest": str((out_dir / "random_forest.pkl").resolve()) if rf is not None else None,
        "pairnet": str((out_dir / "pairnet.pt").resolve()) if best_nn_model is not None else None,
        "graph_reranker": str((out_dir / "graph_reranker.pt").resolve()) if best_graph_model is not None else None,
    }
    final_metrics = {
        "model_variant": model_variant,
        "model_label": model_label,
        "base_edge_variant": base_edge_variant,
        "output_dir": str(out_dir.resolve()),
        "device": device,
        "device_name": torch.cuda.get_device_name(0),
        "train_image_count": len(train_images),
        "test_image": str(test_image_path),
        "reference_source": reference_source,
        "rows": args.rows,
        "cols": args.cols,
        "strip_size": args.strip_size,
        "negatives_per_positive": args.negatives_per_positive,
        "epochs": best_metrics["epochs"],
        "epoch_step": epoch_step,
        "max_epochs": max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "rf_weight": args.rf_weight,
        "graph_weight": args.graph_weight,
        "feature_mode": args.feature_mode,
        "cnn_input_mode": args.cnn_input_mode,
        "rf_jobs": args.rf_jobs,
        "val_ratio": args.val_ratio,
        "graph_candidate_k": args.graph_candidate_k,
        "graph_train_image_limit": args.graph_train_image_limit,
        "graph_val_image_limit": args.graph_val_image_limit,
        "graph_model_type": args.graph_model_type,
        "graph_hidden_dim": args.graph_hidden_dim,
        "graph_layers": args.graph_layers,
        "graph_epochs": args.graph_epochs,
        "graph_learning_rate": args.graph_lr,
        "graph_hard_negative_ratio": args.graph_hard_negative_ratio,
        "initial_pair_candidates": args.initial_pair_candidates,
        "beam_width": args.beam_width,
        "beam_branch_factor": args.beam_branch_factor,
        "border_bonus_weight": args.border_bonus_weight,
        "anneal_steps": args.anneal_steps,
        "global_search_restarts": args.global_search_restarts,
        "global_search_score_noise": args.global_search_score_noise,
        "search_candidates_evaluated": best_metrics["search_candidates_evaluated"],
        "best_arrangement_score": best_metrics["best_arrangement_score"],
        "stain_normalization": "none" if stain_normalizer is None else stain_normalizer.method,
        "stain_normalizer_reference_images": 0 if stain_normalizer is None else stain_normalizer.reference_image_count,
        "stain_normalizer_reference_pixels": 0 if stain_normalizer is None else stain_normalizer.reference_pixel_count,
        "used_existing_slices_for_training": split_metadata["data_source"] == "existing_slice_folders",
        "used_existing_slices_for_reference": args.use_existing_slices,
        "training_data_source": split_metadata["data_source"],
        "validation_split_strategy": split_metadata["split_strategy"],
        "train_split_images": split_metadata["train_split_images"],
        "val_split_images": split_metadata["val_split_images"],
        "allow_flips": using_randomized_slice_pieces or not args.no_flips,
        "allow_rotations": using_randomized_slice_pieces,
        "orientation_state_count": len(orientation_state_values),
        "graph_orientation_state_count": len(graph_state_values),
        "used_randomized_slice_pieces": using_randomized_slice_pieces,
        "position_accuracy": best_metrics["position_accuracy"],
        "adjacency_accuracy": best_metrics["adjacency_accuracy"],
        "orientation_accuracy": best_metrics["orientation_accuracy"],
        "joint_position_orientation_accuracy": best_metrics["joint_position_orientation_accuracy"],
        "ssim": best_metrics["ssim"],
        "pixel_match_ratio": best_metrics["pixel_match_ratio"],
        "model_exact_match": best_metrics["exact_match"],
        "exact_match_reached": exact_match_reached,
        "reference_exact_match": reference_exact_match,
        "graph_validation_auc": None if best_graph_validation is None else best_graph_validation["auc"],
        "graph_validation_ap": None if best_graph_validation is None else best_graph_validation["ap"],
        "graph_validation_acc": None if best_graph_validation is None else best_graph_validation["acc"],
        "exported_reconstruction_source": "model_prediction",
        "history": history,
        "timings": {
            "random_forest_seconds": rf_seconds,
            "pairnet_seconds_total": total_nn_seconds,
            "graph_reranker_seconds_total": total_graph_seconds,
            "reconstruction_seconds_total": total_recon_seconds,
            "total_seconds": time.perf_counter() - overall_start,
        },
        "artifacts": artifacts,
    }

    save_experiment_outputs(
        out_dir,
        reference_img,
        scrambled_img,
        best_reconstructed_img,
        final_metrics,
        rf,
        best_nn_model,
        best_graph_model,
    )
    save_single_run_summary_table(out_dir, final_metrics)
    status_msg = (
        f"\n[{model_label}] reconstruction reached an exact match at {best_metrics['epochs']} epochs."
        if exact_match_reached
        else (
            f"\n[{model_label}] reconstruction did not reach an exact match by {max_epochs} epochs; "
            f"saved the best result at {best_metrics['epochs']} epochs."
        )
    )
    print(status_msg)
    print(f"Saved outputs to: {out_dir.resolve()}")
    return final_metrics


def run_batch_experiment(args) -> Dict[str, object]:
    base_dir = Path(args.train_dir) if args.train_dir else get_default_image_dir()
    image_paths = list_image_paths(base_dir)
    if args.batch_image_limit > 0:
        image_paths = image_paths[:args.batch_image_limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found for batch evaluation in {base_dir}")

    batch_root = Path(args.output_dir)
    batch_rows: List[Dict[str, object]] = []

    for index, test_image_path in enumerate(image_paths, start=1):
        run_folder = batch_root / f"{index:03d}_{test_image_path.stem}"
        run_args = clone_run_args(
            args,
            test_image=str(test_image_path),
            output_dir=str(run_folder),
            batch_eval_all_images=False,
        )
        print(f"\n[Batch Image {index}/{len(image_paths)}] {test_image_path.name}")
        metrics = run_experiment(run_args)
        batch_rows.append(flatten_metrics_for_table(metrics))

    summary_rows = list(batch_rows)
    summary_rows.append(aggregate_table_rows(batch_rows))
    summary_csv = write_metrics_rows_csv(batch_root / "image_level_experiment_summary.csv", summary_rows)
    print(f"Saved batch image summary to: {summary_csv.resolve()}")
    return {
        "output_dir": str(batch_root.resolve()),
        "image_count": len(batch_rows),
        "summary_csv": str(summary_csv.resolve()),
    }


def run_from_args(args: argparse.Namespace) -> Dict[str, object]:
    if getattr(args, "batch_eval_all_images", False):
        return run_batch_experiment(args)
    return run_experiment(args)


def build_argparser():
    parser = argparse.ArgumentParser(description="Pathology microscopy tile reconstruction experiment")
    for flags, kwargs in (
        (("--train_dir",), {"type": str, "default": None, "help": "Folder with training pathology images. Defaults to ./image_file when available."}),
        (("--test_image",), {"type": str, "default": None, "help": "A single test image for scramble/reconstruction. Defaults to the last image found."}),
        (("--output_dir",), {"type": str, "default": "./outputs"}),
        (("--grid_choice",), {"type": str, "default": None, "choices": tuple(GRID_CHOICE_PRESETS), "help": "Preset grid selector: 1=3x6, 2=4x5, 3=5x4, 4=5x7, 5=10x8, 6=5x5."}),
        (("--rows",), {"type": int, "default": 4, "help": "Grid rows"}),
        (("--cols",), {"type": int, "default": 5, "help": "Grid cols"}),
        (("--strip_size",), {"type": int, "default": 16}),
        (("--negatives_per_positive",), {"type": int, "default": 2}),
        (("--epochs",), {"type": int, "default": 20, "help": "Starting epoch count for the progressive training loop."}),
        (("--epoch_step",), {"type": int, "default": 20, "help": "How many epochs to add on each retry."}),
        (("--max_epochs",), {"type": int, "default": 200, "help": "Maximum total epochs to try before stopping."}),
        (("--batch_size",), {"type": int, "default": 64}),
        (("--lr",), {"type": float, "default": 1e-3}),
        (("--rf_jobs",), {"type": int, "default": 1, "help": "RandomForest parallel jobs. Use 1 for the most compatible Windows/sandbox behavior."}),
        (("--rf_weight",), {"type": float, "default": 0.5, "help": "Fusion weight for RF; NN weight = 1 - rf_weight"}),
        (("--graph_weight",), {"type": float, "default": 0.0, "help": "Blend weight for the graph edge reranker; 0 disables it, 1 uses only graph-corrected candidate edges."}),
        (("--feature_mode",), {"type": str, "default": "fusion", "choices": FEATURE_MODES, "help": "Handcrafted pathology feature family used by the RandomForest branch."}),
        (("--cnn_input_mode",), {"type": str, "default": "rgb_hed", "choices": ("rgb", "rgb_hed"), "help": "CNN strip representation. 'rgb_hed' adds H&E stain channels to RGB."}),
        (("--stain_norm",), {"type": str, "default": "none", "choices": ("none", "reinhard"), "help": "Optional stain normalization method applied before RF/CNN/graph feature extraction."}),
        (("--stain_norm_fit_images",), {"type": int, "default": 16, "help": "How many training images are sampled to fit the stain normalization reference profile."}),
        (("--stain_norm_fit_pixels",), {"type": int, "default": 8192, "help": "Maximum tissue pixels sampled per image when fitting the stain normalization reference profile."}),
        (("--val_ratio",), {"type": float, "default": 0.25, "help": "Image-level validation split ratio before pair generation."}),
        (("--slice_root",), {"type": str, "default": str(default_slice_root()), "help": "Root directory that contains folders like pathology_001_5x4, singlepage_001_5x4, ..."}),
        (("--use_existing_slices",), {"dest": "use_existing_slices", "action": "store_true", "help": "Use existing slice folders for training/reference when available."}),
        (("--no_existing_slices",), {"dest": "use_existing_slices", "action": "store_false", "help": "Disable the existing-slice-folder path and fall back to full-image tiling."}),
        (("--initial_pair_candidates",), {"type": int, "default": 16, "help": "Number of top-scoring initial pair candidates to try during reconstruction."}),
        (("--anneal_steps",), {"type": int, "default": 4000, "help": "Simulated-annealing steps applied after greedy reconstruction."}),
        (("--beam_width",), {"type": int, "default": 64, "help": "Beam width used before simulated annealing."}),
        (("--beam_branch_factor",), {"type": int, "default": 16, "help": "Top candidate expansions kept per beam state at each placement step."}),
        (("--global_search_restarts",), {"type": int, "default": 1, "help": "How many diversified beam+anneal reconstruction restarts to evaluate per epoch."}),
        (("--global_search_score_noise",), {"type": float, "default": 0.02, "help": "Small score perturbation used to diversify global-search restarts."}),
        (("--border_bonus_weight",), {"type": float, "default": 0.35, "help": "Bonus weight that favors placing likely outer-border tile edges on the puzzle boundary."}),
        (("--graph_candidate_k",), {"type": int, "default": 12, "help": "Top-K candidate adjacency edges kept per source-state and relation before graph reranking."}),
        (("--graph_train_image_limit",), {"type": int, "default": 0, "help": "Optional limit on how many training images are converted into graph samples. 0 uses the full split."}),
        (("--graph_val_image_limit",), {"type": int, "default": 0, "help": "Optional limit on how many validation images are converted into graph samples. 0 uses the full split."}),
        (("--graph_model_type",), {"type": str, "default": "graphsage", "choices": ("graphsage", "gat"), "help": "Graph reranker backbone. 'graphsage' uses mean aggregation, while 'gat' applies attention-weighted aggregation."}),
        (("--graph_hidden_dim",), {"type": int, "default": 128, "help": "Hidden dimension for the graph reranker."}),
        (("--graph_layers",), {"type": int, "default": 3, "help": "Message-passing layer count for the graph reranker."}),
        (("--graph_epochs",), {"type": int, "default": 8, "help": "Training epochs for the graph reranker."}),
        (("--graph_lr",), {"type": float, "default": 1e-3, "help": "Learning rate for the graph reranker."}),
        (("--graph_hard_negative_ratio",), {"type": float, "default": 8.0, "help": "How many high-scoring negatives to keep per positive edge when training the graph reranker."}),
        (("--batch_eval_all_images",), {"action": "store_true", "help": "Run leave-one-image-out style reconstruction for every image in train_dir and export an image-level summary CSV."}),
        (("--batch_image_limit",), {"type": int, "default": 0, "help": "Optional limit on how many images are included in batch image-level evaluation. 0 uses every image."}),
        (("--no_flips",), {"action": "store_true", "help": "Disable random flips when scrambling the test image."}),
        (("--cpu",), {"action": "store_true", "help": "Deprecated. CPU mode is now disabled because training must use GPU."}),
    ):
        parser.add_argument(*flags, **kwargs)
    parser.set_defaults(use_existing_slices=True)
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, object]:
    args = parse_cli_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    main(sys.argv[1:])
