#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests


def log(message: str) -> None:
    print(message, flush=True)


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_json_env(name: str) -> dict[str, Any]:
    value = env(name)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def coerce_scalar(value: str) -> Any:
    lower = value.strip().lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    if lower == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def add_api_key(url: str, api_key: str | None) -> str:
    if not api_key:
        return url
    parsed = urlparse(url)
    if parsed.netloc != "api.roboflow.com":
        return url
    query = parse_qs(parsed.query, keep_blank_values=True)
    if "api_key" not in query:
        query["api_key"] = [api_key]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def parse_roboflow_ref(ref: str) -> tuple[str, str, str] | None:
    value = ref.strip()
    if not value:
        return None

    if "://" not in value:
        parts = [part for part in value.strip("/").split("/") if part]
    else:
        parsed = urlparse(value)
        parts = [part for part in parsed.path.strip("/").split("/") if part]

    if len(parts) == 3 and parts[2].isdigit():
        return parts[0], parts[1], parts[2]

    for marker in ("dataset", "version"):
        if marker in parts:
            index = parts.index(marker)
            if index >= 2 and index + 1 < len(parts):
                return parts[index - 2], parts[index - 1], parts[index + 1]

    for index, part in enumerate(parts):
        if part.isdigit() and index >= 2:
            return parts[index - 2], parts[index - 1], part

    return None


def is_direct_zip_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    path = parsed.path.lower()
    if path.endswith(".zip"):
        return True
    if parsed.netloc.endswith("roboflow.com") and path.startswith("/ds/"):
        return True
    return False


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\rDownloading dataset: {pct:5.1f}%", end="", flush=True)
    if total:
        print(flush=True)


def extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def find_dataset_root(base_dir: Path) -> Path:
    candidates = [base_dir]
    candidates.extend(path for path in base_dir.iterdir() if path.is_dir())

    for candidate in candidates:
        coco_ann = candidate / "train" / "_annotations.coco.json"
        yolo_yaml = candidate / "data.yaml"
        yolo_train = candidate / "train" / "images"
        if coco_ann.exists() or (yolo_yaml.exists() and yolo_train.exists()):
            return candidate

    return base_dir


def validate_dataset_root(dataset_root: Path) -> None:
    coco_ann = dataset_root / "train" / "_annotations.coco.json"
    yolo_yaml = dataset_root / "data.yaml"
    yolo_train = dataset_root / "train" / "images"
    if coco_ann.exists() or (yolo_yaml.exists() and yolo_train.exists()):
        return

    raise RuntimeError(
        "Downloaded dataset is not in a trainable RF-DETR layout. Expected either "
        f"{coco_ann} for COCO or {yolo_yaml} plus {yolo_train} for YOLO. "
        "Try DATASET_FORMAT=coco for detection or DATASET_FORMAT=coco-segmentation for segmentation."
    )


def download_with_direct_url(url: str, dataset_dir: Path, api_key: str | None) -> Path:
    clear_directory(dataset_dir)
    zip_path = dataset_dir / "roboflow.zip"
    download_file(add_api_key(url, api_key), zip_path)
    extract_zip(zip_path, dataset_dir)
    zip_path.unlink(missing_ok=True)
    return find_dataset_root(dataset_dir)


def download_with_api(
    workspace: str,
    project: str,
    version: str,
    dataset_format: str,
    dataset_dir: Path,
    api_key: str,
) -> Path:
    clear_directory(dataset_dir)

    from roboflow import Roboflow

    log(f"Downloading Roboflow dataset {workspace}/{project}/{version} as {dataset_format}...")
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(int(version)).download(
        model_format=dataset_format,
        location=str(dataset_dir),
        overwrite=True,
    )
    location = Path(getattr(dataset, "location", dataset_dir))
    return find_dataset_root(location)


def resolve_dataset() -> Path:
    dataset_dir = Path(env("DATASET_DIR", "/workspace/dataset")).expanduser()
    dataset_format = env("DATASET_FORMAT", "coco")
    api_key = env("ROBOFLOW_API_KEY")
    dataset_url = env("ROBOFLOW_DATASET_URL")

    if env_bool("SKIP_DATASET_DOWNLOAD", False):
        log("SKIP_DATASET_DOWNLOAD=true, using existing DATASET_DIR.")
        return find_dataset_root(dataset_dir)

    if dataset_url and is_direct_zip_url(dataset_url):
        log("Downloading dataset from direct Roboflow ZIP URL...")
        return download_with_direct_url(dataset_url, dataset_dir, api_key)

    workspace = env("ROBOFLOW_WORKSPACE")
    project = env("ROBOFLOW_PROJECT")
    version = env("ROBOFLOW_VERSION")

    if dataset_url:
        parsed = parse_roboflow_ref(dataset_url)
        if parsed:
            workspace, project, version = parsed

    if not workspace or not project or not version:
        raise RuntimeError(
            "Set ROBOFLOW_DATASET_URL to a Roboflow dataset URL, or set "
            "ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, and ROBOFLOW_VERSION."
        )
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY as a runtime environment variable.")

    return download_with_api(workspace, project, version, dataset_format, dataset_dir, api_key)


def get_model_class() -> type[Any]:
    import rfdetr

    task = env("MODEL_TASK", "detection").strip().lower()
    variant = env("MODEL_VARIANT", "medium").strip().lower().replace("-", "").replace("_", "")

    detection = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
        "xlarge": "RFDETRXLarge",
        "xl": "RFDETRXLarge",
        "2xlarge": "RFDETR2XLarge",
        "2xl": "RFDETR2XLarge",
    }
    segmentation = {
        "nano": "RFDETRSegNano",
        "small": "RFDETRSegSmall",
        "medium": "RFDETRSegMedium",
        "large": "RFDETRSegLarge",
        "xlarge": "RFDETRSegXLarge",
        "xl": "RFDETRSegXLarge",
        "2xlarge": "RFDETRSeg2XLarge",
        "2xl": "RFDETRSeg2XLarge",
    }

    if task == "detection":
        class_name = detection.get(variant)
    elif task == "segmentation":
        class_name = segmentation.get(variant)
    else:
        raise ValueError("MODEL_TASK must be 'detection' or 'segmentation'")

    if not class_name or not hasattr(rfdetr, class_name):
        raise ValueError(f"Unsupported RF-DETR model: task={task!r}, variant={variant!r}")
    return getattr(rfdetr, class_name)


def build_train_args(dataset_root: Path) -> dict[str, Any]:
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    args: dict[str, Any] = {
        "dataset_dir": str(dataset_root),
        "output_dir": str(output_dir),
    }

    direct_env_map = {
        "EPOCHS": "epochs",
        "BATCH_SIZE": "batch_size",
        "GRAD_ACCUM_STEPS": "grad_accum_steps",
        "LR": "lr",
        "LR_ENCODER": "lr_encoder",
        "WEIGHT_DECAY": "weight_decay",
        "NUM_WORKERS": "num_workers",
        "DEVICE": "device",
        "RESUME": "resume",
        "PROJECT": "project",
        "RUN": "run",
    }
    for env_name, arg_name in direct_env_map.items():
        value = env(env_name)
        if value is not None:
            args[arg_name] = coerce_scalar(value)

    bool_env_map = {
        "EARLY_STOPPING": "early_stopping",
        "TENSORBOARD": "tensorboard",
        "WANDB": "wandb",
        "USE_EMA": "use_ema",
        "RUN_TEST": "run_test",
        "GRADIENT_CHECKPOINTING": "gradient_checkpointing",
    }
    for env_name, arg_name in bool_env_map.items():
        value = env(env_name)
        if value is not None:
            args[arg_name] = env_bool(env_name)

    args.update(parse_json_env("EXTRA_TRAIN_ARGS_JSON"))
    return args


def get_output_dir() -> Path:
    return Path(env("OUTPUT_DIR", "/workspace/output")).expanduser()


def get_completion_file(output_dir: Path) -> Path:
    return Path(env("TRAINING_COMPLETE_FILE", str(output_dir / ".training_complete"))).expanduser()


def exit_if_training_already_completed() -> None:
    output_dir = get_output_dir()
    completion_file = get_completion_file(output_dir)
    if env_bool("FORCE_RERUN", False):
        return
    if not completion_file.exists():
        return

    log(f"Training already completed; found {completion_file}.")
    log("Set FORCE_RERUN=true or change OUTPUT_DIR to start another training run.")
    keep_alive_after_training()
    raise SystemExit(0)


def mark_training_completed(output_dir: Path) -> None:
    completion_file = get_completion_file(output_dir)
    completion_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
    }
    completion_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def keep_alive_after_training() -> None:
    if not env_bool("KEEP_ALIVE_AFTER_TRAINING", True):
        return

    shell = env("POST_TRAINING_SHELL", "/bin/bash")
    if shell and sys.stdin.isatty() and shutil.which(shell):
        log(f"Training finished. Opening interactive shell: {shell}")
        os.execvp(shell, [shell])

    log("Training finished. Keeping container idle; attach with a terminal or stop the pod when done.")
    while True:
        time.sleep(3600)


def main() -> int:
    exit_if_training_already_completed()

    dataset_root = resolve_dataset()
    validate_dataset_root(dataset_root)
    log(f"Using dataset directory: {dataset_root}")

    model_class = get_model_class()
    model_kwargs = parse_json_env("MODEL_INIT_JSON")
    log(f"Creating model: {model_class.__name__}")
    model = model_class(**model_kwargs)

    train_args = build_train_args(dataset_root)
    log("Starting training with arguments:")
    log(json.dumps({k: str(v) for k, v in train_args.items()}, indent=2, sort_keys=True))
    model.train(**train_args)

    if env_bool("EXPORT_ONNX", False):
        export_args = parse_json_env("EXTRA_EXPORT_ARGS_JSON")
        output_dir = train_args["output_dir"]
        log(f"Exporting ONNX to {output_dir}...")
        model.export(output_dir=output_dir, **export_args)

    mark_training_completed(Path(train_args["output_dir"]))
    log("Training job complete.")
    keep_alive_after_training()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
