# RF-DETR RunPod Training Image

Reusable Docker image for training Roboflow RF-DETR models on GPU machines such as RunPod.

The container downloads a Roboflow dataset at runtime, validates the dataset layout, and then calls `model.train(...)`. Your Roboflow API key is supplied as a runtime environment variable and is never baked into the image.

## Use The Container

Use this Docker image in your RunPod template:

```text
petemaher/rfdetr-train:1.0.0
```

Set the container command to the default command from the image. The image runs:

```bash
python /opt/rfdetr/train_rfdetr.py
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
BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
LR=1e-4
OUTPUT_DIR=/workspace/output
DATASET_DIR=/workspace/dataset
FORCE_RERUN=false
TENSORBOARD=true
WANDB=false
```

Supported `MODEL_VARIANT` values for detection are `nano`, `small`, `medium`, `large`, `xlarge`, and `2xlarge`.

Supported `MODEL_TASK` values are `detection` and `segmentation`. For segmentation, set `DATASET_FORMAT=coco-segmentation`.

`DATASET_FORMAT=coco` is the safest default for object detection because RF-DETR detects COCO data from `train/_annotations.coco.json`. YOLO can work too with `DATASET_FORMAT=yolov5pytorch`, where RF-DETR detects `data.yaml` plus `train/images/`.

For less common RF-DETR train parameters, pass JSON:

```bash
EXTRA_TRAIN_ARGS_JSON='{"eval_interval":5,"log_per_class_metrics":false,"pin_memory":true,"persistent_workers":true,"prefetch_factor":4,"patience":10,"checkpoint_interval":5}'
MODEL_INIT_JSON='{"pretrain_weights": null}'
EXTRA_EXPORT_ARGS_JSON='{"dynamic_batch": true}'
```

## RunPod Setup

Create a RunPod pod or template with:

```text
Container image: petemaher/rfdetr-train:1.0.0
Container disk:  50 GB
Volume disk:     150 GB or larger
Volume mount:    /workspace
```

Start with:

```bash
ROBOFLOW_API_KEY=...
ROBOFLOW_DATASET_URL=...
DATASET_FORMAT=coco
MODEL_VARIANT=medium
EPOCHS=100
BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
LR=1e-4
OUTPUT_DIR=/workspace/output
EXTRA_TRAIN_ARGS_JSON={"eval_interval":5,"log_per_class_metrics":false,"pin_memory":true,"persistent_workers":true,"prefetch_factor":4}
```

Use a GPU with enough VRAM for the selected variant. Start with `MODEL_VARIANT=medium`, `BATCH_SIZE=8`, and `GRAD_ACCUM_STEPS=2`; if VRAM is still underused, try `BATCH_SIZE=16` and `GRAD_ACCUM_STEPS=1`. If you hit out-of-memory, lower `BATCH_SIZE` or use `BATCH_SIZE=auto`.

Mount persistent storage at `/workspace` or at least `/workspace/output` so checkpoints survive the pod lifecycle.

Checkpoints are written to `OUTPUT_DIR`, including RF-DETR's best checkpoint files such as `checkpoint_best_total.pth`.

When training completes successfully, the container writes:

```text
/workspace/output/.training_complete
```

If the training command is invoked again with the same `OUTPUT_DIR`, the script exits without starting another run. To intentionally train again, set a new `OUTPUT_DIR` or set:

```bash
FORCE_RERUN=true
```

## Local GPU Test

```bash
docker run --rm --gpus all --ipc=host \
  -e ROBOFLOW_API_KEY="$ROBOFLOW_API_KEY" \
  -e ROBOFLOW_DATASET_URL="https://universe.roboflow.com/workspace-slug/project-slug/dataset/1" \
  -e EPOCHS=10 \
  -e BATCH_SIZE=8 \
  -e GRAD_ACCUM_STEPS=2 \
  -e EXTRA_TRAIN_ARGS_JSON='{"eval_interval":5,"log_per_class_metrics":false,"pin_memory":true,"persistent_workers":true,"prefetch_factor":4}' \
  -v "$PWD/output:/workspace/output" \
  petemaher/rfdetr-train:1.0.0
```

## Output Files

Training output is written to:

```text
/workspace/output
```

The completion guard is written to:

```text
/workspace/output/.training_complete
```

Override the guard path with:

```bash
TRAINING_COMPLETE_FILE=/workspace/output/my-complete-marker.json
```

Dataset downloads are written to:

```text
/workspace/dataset
```

The RF-DETR model cache can be redirected with `RF_HOME` if needed:

```bash
RF_HOME=/workspace/models
```

## Publish New Versions

These steps are for maintainers updating this image.

Build a new image tag:

```bash
docker build -t petemaher/rfdetr-train:1.0.1 .
```

Log in to Docker Hub if needed:

```bash
docker login -u petemaher
```

Push the new image:

```bash
docker push petemaher/rfdetr-train:1.0.1
```

Optionally update the floating `latest` tag:

```bash
docker tag petemaher/rfdetr-train:1.0.1 petemaher/rfdetr-train:latest
docker push petemaher/rfdetr-train:latest
```

If Docker requires `sudo`, log in and push with the same Docker user context:

```bash
sudo docker login -u petemaher
sudo docker push petemaher/rfdetr-train:1.0.1
```

Then commit and push the README or code changes:

```bash
git add .
git commit -m "Update training image docs"
git push origin main
```
