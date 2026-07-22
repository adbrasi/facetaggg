from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import torch

import folder_paths

ULTRALYTICS_DIR = os.path.join(folder_paths.models_dir, "ultralytics")
MODEL_EXTENSIONS = (".pt", ".onnx", ".engine")

_MODEL_CACHE: dict[str, object] = {}


def list_ultralytics_models() -> list[str]:
    models = []
    if os.path.isdir(ULTRALYTICS_DIR):
        for root, _dirs, files in os.walk(ULTRALYTICS_DIR):
            for name in files:
                if name.lower().endswith(MODEL_EXTENSIONS):
                    relative = os.path.relpath(os.path.join(root, name), ULTRALYTICS_DIR)
                    models.append(relative.replace(os.sep, "/"))
    return sorted(models) or ["(no models found in models/ultralytics)"]


def _import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        import subprocess
        import sys

        print("[FaceTag] 'ultralytics' not found; installing it now (one-time)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
    return YOLO


def load_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        path = os.path.join(ULTRALYTICS_DIR, model_name.replace("/", os.sep))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"FaceTag: model not found: {path}")
        YOLO = _import_yolo()
        _MODEL_CACHE[model_name] = YOLO(path)
    return _MODEL_CACHE[model_name]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def size(self) -> tuple[float, float]:
        return (max(1.0, self.x2 - self.x1), max(1.0, self.y2 - self.y1))


def select_primary(detections: Iterable[Box], previous: Box | None) -> Box | None:
    candidates = list(detections)
    if not candidates:
        return None
    if previous is None:
        return max(candidates, key=lambda box: box.size[0] * box.size[1] * box.confidence)
    px, py = previous.center
    scale = max(previous.size)
    return max(
        candidates,
        key=lambda box: box.confidence / (1.0 + np.hypot(box.center[0] - px, box.center[1] - py) / scale),
    )


def fill_missing(boxes: list[Box | None], max_gap: int, frame_size: tuple[int, int]) -> tuple[list[Box], bool]:
    """Interpolate missing detections; fall back to the full frame instead of failing."""
    if not any(box is not None for box in boxes):
        width, height = frame_size
        full = Box(0.0, 0.0, float(width), float(height), 0.0)
        return [full] * len(boxes), True
    values = np.full((len(boxes), 5), np.nan, dtype=np.float64)
    for index, box in enumerate(boxes):
        if box is not None:
            values[index] = [box.x1, box.y1, box.x2, box.y2, box.confidence]
    known = np.flatnonzero(~np.isnan(values[:, 0]))
    for column in range(values.shape[1]):
        values[:, column] = np.interp(np.arange(len(boxes)), known, values[known, column])
    # Large internal gaps hold the nearest detection instead of sweeping across the frame.
    for left, right in zip(known, known[1:]):
        if right - left - 1 > max_gap:
            midpoint = (left + right) // 2
            values[left + 1 : midpoint + 1] = values[left]
            values[midpoint + 1 : right] = values[right]
    return [Box(*row) for row in values.tolist()], False


def zero_phase_ema(values: np.ndarray, smoothing: float) -> np.ndarray:
    if len(values) < 2 or smoothing <= 0:
        return values.copy()

    def run(items: np.ndarray) -> np.ndarray:
        result = items.copy()
        for index in range(1, len(items)):
            result[index] = smoothing * result[index - 1] + (1 - smoothing) * items[index]
        return result

    return (run(values) + run(values[::-1])[::-1]) / 2


def build_crops(
    boxes: list[Box], frame_size: tuple[int, int], output_size: tuple[int, int],
    padding: float, smoothing: float, zoom_mode: str, zoom_percentile: float,
) -> list[tuple[float, float, float, float]]:
    """follow_detection framing: steady camera, crop always kept inside the frame."""
    frame_width, frame_height = frame_size
    aspect = output_size[0] / output_size[1]
    geometry = []
    for box in boxes:
        cx, cy = box.center
        width, height = box.size
        crop_width = max(width * padding, height * padding * aspect, 8.0)
        geometry.append((cx, cy, np.log(crop_width)))
    geometry = np.asarray(geometry, dtype=np.float64)
    smooth = zero_phase_ema(geometry, smoothing)
    if zoom_mode == "fixed":
        fixed_width = float(np.percentile(np.exp(geometry[:, 2]), zoom_percentile))
        smooth[:, 2] = np.log(fixed_width)
    crops = []
    for cx, cy, log_width in smooth:
        width = min(float(np.exp(log_width)), frame_width, frame_height * aspect)
        height = width / aspect
        cx = float(np.clip(cx, width / 2, frame_width - width / 2))
        cy = float(np.clip(cy, height / 2, frame_height - height / 2))
        crops.append((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))
    return crops


def build_crops_locked_center(
    boxes: list[Box], frame_size: tuple[int, int], output_size: tuple[int, int],
    padding: float, smoothing: float,
) -> list[tuple[float, float, float, float]]:
    """locked_center framing: the crop is centered exactly on the subject and its
    size is auto-zoomed so the window always fits inside the frame. No padding,
    no mirror, no edge fill -- when the subject is near a border the crop simply
    zooms in enough to stay on real pixels."""
    frame_width, frame_height = frame_size
    aspect = output_size[0] / output_size[1]
    centers = np.asarray([box.center for box in boxes], dtype=np.float64)
    widths = np.asarray(
        [max(box.size[0] * padding, box.size[1] * padding * aspect, 8.0) for box in boxes],
        dtype=np.float64,
    )
    smooth_centers = zero_phase_ema(centers, smoothing)
    smooth_widths = np.exp(zero_phase_ema(np.log(widths), smoothing))
    crops = []
    for (cx, cy), desired in zip(smooth_centers, smooth_widths):
        cx = float(np.clip(cx, 1.0, frame_width - 1.0))
        cy = float(np.clip(cy, 1.0, frame_height - 1.0))
        # Largest crop that stays centered on (cx, cy) without leaving the frame.
        max_width = min(
            2.0 * min(cx, frame_width - cx),
            2.0 * aspect * min(cy, frame_height - cy),
            float(frame_width),
            frame_height * aspect,
        )
        width = float(np.clip(desired, min(max_width, 4.0), max(max_width, 4.0)))
        height = width / aspect
        crops.append((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))
    return crops


def crop_with_padding(frame: np.ndarray, crop: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[float, float]]:
    x1, y1, x2, y2 = crop
    width = max(1, int(round(x2 - x1)))
    height = max(1, int(round(y2 - y1)))
    padded = cv2.copyMakeBorder(frame, height, height, width, width, cv2.BORDER_REFLECT_101)
    center = ((x1 + x2) / 2 + width, (y1 + y2) / 2 + height)
    sampled = cv2.getRectSubPix(padded, (width, height), center)
    # getRectSubPix samples symmetrically around the center with a fixed size.
    origin = ((x1 + x2) / 2 - (width - 1) / 2, (y1 + y2) / 2 - (height - 1) / 2)
    return sampled, origin


def detect_batch(
    model, frames: list[np.ndarray], confidence: float, image_size: int,
    device: str, batch_size: int = 16,
) -> list[list[Box]]:
    half = device != "cpu"
    detections: list[list[Box]] = []
    for start in range(0, len(frames), batch_size):
        results = model.predict(
            source=frames[start : start + batch_size],
            conf=confidence,
            device=device,
            imgsz=image_size,
            half=half,
            verbose=False,
        )
        for result in results:
            detections.append(
                [Box(*map(float, box.xyxy[0].tolist()), float(box.conf.item())) for box in result.boxes]
                if result.boxes is not None
                else []
            )
    return detections


class FaceTagCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (list_ultralytics_models(),),
                "confidence": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "padding": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05,
                                      "tooltip": "Crop size as a multiple of the detected box"}),
                "output_width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "output_height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "smoothing": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 0.99, "step": 0.01,
                                        "tooltip": "Temporal smoothing across the batch (video frames)"}),
                "zoom_mode": (["fixed", "smooth"], {"tooltip": "fixed: one zoom level for the whole batch"}),
                "zoom_percentile": ("FLOAT", {"default": 95.0, "min": 50.0, "max": 100.0, "step": 0.5}),
                "mask_mode": (["follow_detection", "locked_center"],
                              {"tooltip": "follow_detection: steady camera, mask tracks the box inside the crop. "
                                          "locked_center: subject nailed to the center, the crop auto-zooms to stay "
                                          "inside the frame (no padding/mirror; zoom_mode is ignored)"}),
                "max_gap": ("INT", {"default": 12, "min": 0, "max": 300,
                                    "tooltip": "Max missing-frame gap bridged by interpolation"}),
                "mask_feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
                                         "tooltip": "Gaussian blur radius on the mask edges, in pixels"}),
                "detector_imgsz": ("INT", {"default": 640, "min": 256, "max": 1536, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "FACETAG_PASTE")
    RETURN_NAMES = ("image", "mask", "preview", "paste_data")
    FUNCTION = "run"
    CATEGORY = "image/facetag"

    def run(
        self, image: torch.Tensor, model_name: str, confidence: float, padding: float,
        output_width: int, output_height: int, smoothing: float, zoom_mode: str,
        zoom_percentile: float, mask_mode: str, max_gap: int, mask_feather: int,
        detector_imgsz: int,
    ):
        frames_rgb = (image.cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames_rgb]
        frame_height, frame_width = frames_bgr[0].shape[:2]

        device = "0" if torch.cuda.is_available() else "cpu"
        model = load_model(model_name)
        detections = detect_batch(model, frames_bgr, confidence, detector_imgsz, device)

        selected: list[Box | None] = []
        previous = None
        for items in detections:
            current = select_primary(items, previous)
            selected.append(current)
            if current is not None:
                previous = current
        boxes, used_fallback = fill_missing(selected, max_gap, (frame_width, frame_height))
        if used_fallback:
            print(f"[FaceTag] no detections in the whole batch (model={model_name}); using full-frame fallback")

        output_size = (output_width, output_height)
        if mask_mode == "locked_center" and not used_fallback:
            # Subject nailed to the center; the crop auto-zooms to stay inside the
            # frame. No borders are ever sampled, so there is no mirror/edge fill.
            crops = build_crops_locked_center(
                boxes, (frame_width, frame_height), output_size, padding, smoothing,
            )
        else:
            crops = build_crops(
                boxes, (frame_width, frame_height), output_size,
                padding, smoothing, zoom_mode, zoom_percentile,
            )

        # In locked_center mode the mask size still breathes with the detection,
        # so smooth it with the same filter used for the camera.
        mask_sizes = None
        if mask_mode == "locked_center" and not used_fallback:
            log_sizes = np.log(np.asarray([box.size for box in boxes], dtype=np.float64))
            mask_sizes = np.exp(zero_phase_ema(log_sizes, smoothing))

        out_images = np.empty((len(frames_bgr), output_height, output_width, 3), dtype=np.float32)
        out_masks = np.empty((len(frames_bgr), output_height, output_width), dtype=np.float32)
        out_previews = np.empty_like(out_images)
        placements: list[tuple[float, float, int, int]] = []
        for index, (frame, box, crop) in enumerate(zip(frames_bgr, boxes, crops)):
            sampled, origin = crop_with_padding(frame, crop)
            # Where this crop sits in the source frame, for pasting back later.
            placements.append((origin[0], origin[1], sampled.shape[1], sampled.shape[0]))
            output = cv2.resize(sampled, output_size, interpolation=cv2.INTER_LANCZOS4)

            if used_fallback:
                mask = np.ones((output_height, output_width), dtype=np.float32)
            elif mask_sizes is not None:
                crop_width = max(crop[2] - crop[0], 1.0)
                crop_height = max(crop[3] - crop[1], 1.0)
                mask_width = float(mask_sizes[index][0]) / crop_width * output_width
                mask_height = float(mask_sizes[index][1]) / crop_height * output_height
                raw_mask = np.zeros((output_height, output_width), dtype=np.uint8)
                x1 = int(round((output_width - mask_width) / 2))
                y1 = int(round((output_height - mask_height) / 2))
                x2 = int(round((output_width + mask_width) / 2))
                y2 = int(round((output_height + mask_height) / 2))
                cv2.rectangle(raw_mask, (x1, y1), (x2, y2), 255, thickness=-1)
                mask = raw_mask.astype(np.float32) / 255.0
            else:
                raw_mask = np.zeros(sampled.shape[:2], dtype=np.uint8)
                x1 = int(round(box.x1 - origin[0]))
                y1 = int(round(box.y1 - origin[1]))
                x2 = int(round(box.x2 - origin[0]))
                y2 = int(round(box.y2 - origin[1]))
                cv2.rectangle(raw_mask, (x1, y1), (x2, y2), 255, thickness=-1)
                mask = cv2.resize(raw_mask, output_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            if mask_feather > 0:
                kernel = mask_feather * 2 + 1
                mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)

            preview = output.astype(np.float32)
            alpha = (mask * 0.35)[..., None]
            green = np.array([60.0, 220.0, 60.0], dtype=np.float32)
            preview = preview * (1.0 - alpha) + green * alpha
            contours, _ = cv2.findContours(
                (mask >= 0.5).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            preview = preview.astype(np.uint8)
            cv2.drawContours(preview, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

            out_images[index] = cv2.cvtColor(output, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out_masks[index] = mask
            out_previews[index] = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        paste_data = {
            "originals": image.cpu(),
            "placements": placements,
            "output_size": output_size,
            "frame_size": (frame_width, frame_height),
        }
        return (
            torch.from_numpy(out_images),
            torch.from_numpy(out_masks),
            torch.from_numpy(out_previews),
            paste_data,
        )


INTERPOLATIONS = {
    "lanczos": cv2.INTER_LANCZOS4,
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
}


def _batch_item(tensor: torch.Tensor, index: int) -> np.ndarray:
    array = tensor[min(index, tensor.shape[0] - 1)].cpu().numpy()
    return array


def _color_transfer(patch: np.ndarray, reference: np.ndarray, weight: np.ndarray, method: str, strength: float) -> np.ndarray:
    """Nudge the pasted patch towards the color statistics of what it covers,
    so an inpaint that shifted hue/exposure blends back in seamlessly."""
    valid = weight > 0.05
    if strength <= 0.0 or valid.sum() < 16:
        return patch
    out = patch.copy()
    if method == "mean_std":
        for channel in range(3):
            src = patch[..., channel][valid]
            ref = reference[..., channel][valid]
            src_std = float(src.std()) + 1e-6
            adjusted = (patch[..., channel] - float(src.mean())) / src_std * (float(ref.std()) + 1e-6) + float(ref.mean())
            out[..., channel] = patch[..., channel] * (1.0 - strength) + adjusted * strength
    elif method == "histogram":
        for channel in range(3):
            src = (patch[..., channel][valid] * 255.0).astype(np.uint8)
            ref = (reference[..., channel][valid] * 255.0).astype(np.uint8)
            src_hist = np.bincount(src, minlength=256).astype(np.float64)
            ref_hist = np.bincount(ref, minlength=256).astype(np.float64)
            src_cdf = np.cumsum(src_hist) / max(src_hist.sum(), 1.0)
            ref_cdf = np.cumsum(ref_hist) / max(ref_hist.sum(), 1.0)
            lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.float32) / 255.0
            mapped = lut[(patch[..., channel] * 255.0).clip(0, 255).astype(np.uint8)]
            out[..., channel] = patch[..., channel] * (1.0 - strength) + mapped * strength
    return np.clip(out, 0.0, 1.0)


class FaceTagPaste:
    """Paste an (inpainted) crop back onto the original frame with a clean blend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paste_data": ("FACETAG_PASTE",),
                "image": ("IMAGE", {"tooltip": "The (inpainted) crop to paste back"}),
                "blend_mode": (["normal", "seamless", "seamless_mixed"],
                               {"tooltip": "seamless uses Poisson blending (cv2.seamlessClone)"}),
                "feather": ("INT", {"default": 12, "min": 0, "max": 512, "step": 1,
                                    "tooltip": "Soft mask edge, in source pixels"}),
                "mask_expand": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1,
                                        "tooltip": "Grow (+) or shrink (-) the paste region before feathering"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_match": (["none", "mean_std", "histogram"],
                                {"tooltip": "Match the pasted colors to what they cover"}),
                "color_match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "interpolation": (list(INTERPOLATIONS.keys()),),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Region to paste (aligned to the crop). "
                                             "If omitted, the whole crop is pasted."}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "background": ("IMAGE", {"tooltip": "Optional base frames to paste onto (same resolution as "
                                                    "the source). Use this to supply a longer/extended timeline; "
                                                    "if omitted, the original frames are used and the last one is "
                                                    "held for any extra inpainted frames."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image/facetag"

    def run(
        self, paste_data: dict, image: torch.Tensor, blend_mode: str, feather: int,
        mask_expand: int, opacity: float, color_match: str, color_match_strength: float,
        interpolation: str, mask: torch.Tensor | None = None, invert_mask: bool = False,
        background: torch.Tensor | None = None,
    ):
        placements = paste_data["placements"]
        interp = INTERPOLATIONS[interpolation]
        base_frames: torch.Tensor = background if background is not None else paste_data["originals"]
        base_len = base_frames.shape[0]
        place_len = len(placements)

        # The output timeline follows the (possibly extended) inpainted sequence.
        # When it is longer than what the crop captured, hold the last background
        # frame and last crop placement for the extra frames (inpaint + extend).
        count = image.shape[0]
        if count != base_len:
            print(f"[FaceTag] paste: {count} inpainted frames vs {base_len} background frames; "
                  f"holding the last background/placement for the extras")
        results = np.empty((count, base_frames.shape[1], base_frames.shape[2], 3), dtype=np.float32)

        for index in range(count):
            base = base_frames[min(index, base_len - 1)].cpu().numpy().astype(np.float32)
            frame_height, frame_width = base.shape[:2]
            origin_x, origin_y, crop_w, crop_h = placements[min(index, place_len - 1)]
            crop_w, crop_h = int(crop_w), int(crop_h)

            patch = _batch_item(image, index).astype(np.float32)
            patch = cv2.resize(patch, (crop_w, crop_h), interpolation=interp)

            if mask is not None:
                patch_mask = _batch_item(mask, index).astype(np.float32)
                if invert_mask:
                    patch_mask = 1.0 - patch_mask
                patch_mask = cv2.resize(patch_mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            else:
                patch_mask = np.ones((crop_h, crop_w), dtype=np.float32)

            if mask_expand != 0:
                ksize = 2 * abs(mask_expand) + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                patch_mask = cv2.dilate(patch_mask, kernel) if mask_expand > 0 else cv2.erode(patch_mask, kernel)
            if feather > 0:
                ksize = 2 * feather + 1
                patch_mask = cv2.GaussianBlur(patch_mask, (ksize, ksize), 0)
            patch_mask = np.clip(patch_mask, 0.0, 1.0) * float(opacity)

            # Intersection of the crop rectangle with the actual frame.
            px, py = int(round(origin_x)), int(round(origin_y))
            sx = max(0, -px)
            sy = max(0, -py)
            dx = max(0, px)
            dy = max(0, py)
            copy_w = min(crop_w - sx, frame_width - dx)
            copy_h = min(crop_h - sy, frame_height - dy)
            if copy_w <= 0 or copy_h <= 0:
                results[index] = base
                continue

            patch_region = patch[sy:sy + copy_h, sx:sx + copy_w]
            alpha_region = patch_mask[sy:sy + copy_h, sx:sx + copy_w]
            dst_region = base[dy:dy + copy_h, dx:dx + copy_w]

            if color_match != "none":
                patch_region = _color_transfer(
                    patch_region, dst_region, alpha_region, color_match, color_match_strength,
                )

            output = base.copy()
            if blend_mode == "normal":
                alpha = alpha_region[..., None]
                output[dy:dy + copy_h, dx:dx + copy_w] = dst_region * (1.0 - alpha) + patch_region * alpha
            else:
                # Poisson blend the pasted patch into the frame for a seamless result.
                hard = base.copy()
                alpha = alpha_region[..., None]
                hard[dy:dy + copy_h, dx:dx + copy_w] = dst_region * (1.0 - alpha) + patch_region * alpha
                binary = np.zeros((frame_height, frame_width), dtype=np.uint8)
                binary[dy:dy + copy_h, dx:dx + copy_w] = (alpha_region > 0.05).astype(np.uint8) * 255
                ys, xs = np.nonzero(binary)
                if len(xs) == 0:
                    output = hard
                else:
                    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
                    flag = cv2.MIXED_CLONE if blend_mode == "seamless_mixed" else cv2.NORMAL_CLONE
                    src_u8 = cv2.cvtColor((hard * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    dst_u8 = cv2.cvtColor((base * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    try:
                        blended = cv2.seamlessClone(src_u8, dst_u8, binary, center, flag)
                        output = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    except cv2.error:
                        output = hard  # seamlessClone is picky near borders; fall back cleanly

            results[index] = np.clip(output, 0.0, 1.0)

        return (torch.from_numpy(results),)


NODE_CLASS_MAPPINGS = {
    "FaceTagCrop": FaceTagCrop,
    "FaceTagPaste": FaceTagPaste,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceTagCrop": "FaceTag Crop (YOLO)",
    "FaceTagPaste": "FaceTag Paste Back",
}
