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


def load_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        path = os.path.join(ULTRALYTICS_DIR, model_name.replace("/", os.sep))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"FaceTag: model not found: {path}")
        from ultralytics import YOLO

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
    clamp_to_frame: bool = True,
) -> list[tuple[float, float, float, float]]:
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
        width = float(np.exp(log_width))
        if clamp_to_frame:
            width = min(width, frame_width, frame_height * aspect)
        height = width / aspect
        if clamp_to_frame:
            cx = float(np.clip(cx, width / 2, frame_width - width / 2))
            cy = float(np.clip(cy, height / 2, frame_height - height / 2))
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
                                          "locked_center: static centered mask, the video moves and zooms to keep "
                                          "the subject on it (zoom_mode is ignored)"}),
                "max_gap": ("INT", {"default": 12, "min": 0, "max": 300,
                                    "tooltip": "Max missing-frame gap bridged by interpolation"}),
                "mask_feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
                                         "tooltip": "Gaussian blur radius on the mask edges, in pixels"}),
                "detector_imgsz": ("INT", {"default": 640, "min": 256, "max": 1536, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "preview")
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
            # The camera does all the work: it tracks and zooms with the smoothed
            # box so the subject always lands on the static centered mask. Frame
            # clamping is disabled (reflect padding covers the edges) and zoom is
            # forced to follow the box, otherwise the subject would drift off the mask.
            crops = build_crops(
                boxes, (frame_width, frame_height), output_size,
                padding, smoothing, "smooth", zoom_percentile, clamp_to_frame=False,
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
        for index, (frame, box, crop) in enumerate(zip(frames_bgr, boxes, crops)):
            sampled, origin = crop_with_padding(frame, crop)
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

        return (
            torch.from_numpy(out_images),
            torch.from_numpy(out_masks),
            torch.from_numpy(out_previews),
        )


NODE_CLASS_MAPPINGS = {
    "FaceTagCrop": FaceTagCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceTagCrop": "FaceTag Crop (YOLO)",
}
