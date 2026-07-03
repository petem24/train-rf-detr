# RF-DETR RunPod Training Image

This is a reusable Docker template for training Roboflow RF-DETR models on a GPU box such as RunPod.

The image downloads a Roboflow dataset at runtime, then calls `model.train(...)`. Keep your Roboflow API key out of the image and pass it as an environment variable in RunPod.

## Build and Push

```bash
docker login
docker build -t YOUR_DOCKERHUB_USER/rfdetr-train:1.8.3 .
docker push YOUR_DOCKERHUB_USER/rfdetr-train:1.8.3
```

Use this image name in a RunPod template:

```text
YOUR_DOCKERHUB_USER/rfdetr-train:1.8.3
```

## Required Environment Variables

```bash
ROBOFLOW_API_KEY=your_roboflow_key
ROBOFLOW_DATASET_URL=https://universe.roboflow.com/workspace-slug/project-slug/dataset/1
```

You can also use explicit project pieces instead of `ROBOFLOW_DATASET_URL`:

```bash
ROBOFLOW_WORKSPACE=workspace-slug
ROBOFLOW_PROJECT=project-slug
ROBOFLOW_VERSION=1
```

## Common Training Settings

```bash
DATASET_FORMAT=coco
MODEL_TASK=detection
MODEL_VARIANT=medium
EPOCHS=100
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
LR=1e-4
OUTPUT_DIR=/workspace/output
DATASET_DIR=/workspace/dataset
TENSORBOARD=true
WANDB=false
```

Supported `MODEL_VARIANT` values for detection are `nano`, `small`, `medium`, `large`, `xlarge`, and `2xlarge`.

Supported `MODEL_TASK` values are `detection` and `segmentation`. For segmentation, set `DATASET_FORMAT=coco-segmentation`.

`DATASET_FORMAT=coco` is the safest default for object detection because RF-DETR detects COCO data from `train/_annotations.coco.json`. YOLO can work too with `DATASET_FORMAT=yolov5pytorch`, where RF-DETR detects `data.yaml` plus `train/images/`.

For less common RF-DETR train parameters, pass JSON:

```bash
EXTRA_TRAIN_ARGS_JSON='{"patience": 10, "checkpoint_interval": 5}'
MODEL_INIT_JSON='{"pretrain_weights": null}'
EXTRA_EXPORT_ARGS_JSON='{"dynamic_batch": true}'
```

## Local GPU Test

```bash
docker run --rm --gpus all --ipc=host \
  -e ROBOFLOW_API_KEY="$ROBOFLOW_API_KEY" \
  -e ROBOFLOW_DATASET_URL="https://universe.roboflow.com/workspace-slug/project-slug/dataset/1" \
  -e EPOCHS=10 \
  -e BATCH_SIZE=4 \
  -e GRAD_ACCUM_STEPS=4 \
  -v "$PWD/output:/workspace/output" \
  YOUR_DOCKERHUB_USER/rfdetr-train:1.8.3
```

## RunPod Template Notes

Set the container image to your Docker Hub image.

Use a GPU with enough VRAM for the selected variant. Start with `MODEL_VARIANT=medium`, `BATCH_SIZE=4`, and `GRAD_ACCUM_STEPS=4`; if you hit out-of-memory, lower `BATCH_SIZE` or use `BATCH_SIZE=auto`.

Mount persistent storage at `/workspace` or at least `/workspace/output` so checkpoints survive the pod lifecycle.

Set these RunPod environment variables:

```bash
ROBOFLOW_API_KEY=...
ROBOFLOW_DATASET_URL=...
DATASET_FORMAT=coco
MODEL_VARIANT=medium
EPOCHS=100
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
LR=1e-4
OUTPUT_DIR=/workspace/output
```

Checkpoints are written to `OUTPUT_DIR`, including RF-DETR's best checkpoint files such as `checkpoint_best_total.pth`.
