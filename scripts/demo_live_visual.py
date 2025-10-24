import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = str(REPO_ROOT / "config")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from callbacks.viz_base import VizCallbackBase
from config.modifier import dynamically_modify_train_config
from data.utils.types import DataType, ObjDetOutput
from modules.utils.fetch import fetch_data_module, fetch_model_module
from utils.evaluation.prophesee.visualize.vis_utils import (
    LABELMAP_GEN1,
    LABELMAP_GEN4_SHORT,
    draw_bboxes,
)


def _merge_demo_cfg(config: DictConfig) -> DictConfig:
    """Inject default demo settings and allow hydra CLI overrides."""
    OmegaConf.set_struct(config, False)
    default_cfg = OmegaConf.create(
        {
            "display": True,
            "show_ground_truth": True,
            "wait_ms": 1,
            "max_batches": None,
            "output_path": None,
            "fps": 12,
        }
    )
    user_cfg = config.get("demo", OmegaConf.create({}))
    demo_cfg = OmegaConf.merge(default_cfg, user_cfg)
    config.demo = demo_cfg
    return demo_cfg


def ev_repr_to_img(ev_repr: np.ndarray) -> np.ndarray:
    """
    Convert stacked histogram representation (pos/neg bins) into a 3-channel visualization.
    """
    if ev_repr.ndim > 3:
        ev_repr = ev_repr.squeeze(0)
    channels, height, width = ev_repr.shape
    assert channels % 2 == 0 and channels > 0, (
        f"Expected even number of channels (pos/neg bins). Got {channels}."
    )
    ev_repr = ev_repr.reshape(2, channels // 2, height, width)
    img_neg = ev_repr[0].sum(axis=0)
    img_pos = ev_repr[1].sum(axis=0)
    img_diff = img_pos - img_neg
    img = 127 * np.ones((height, width, 3), dtype=np.uint8)
    img[img_diff > 0] = 255
    img[img_diff < 0] = 0
    return img


def _resolve_label_map(dataset_name: str):
    if dataset_name == "gen1":
        return LABELMAP_GEN1
    if dataset_name == "gen4":
        return LABELMAP_GEN4_SHORT
    raise NotImplementedError(f"Unsupported dataset '{dataset_name}' for demo visualisation.")


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    data = batch["data"]

    if DataType.EV_REPR in data:
        data[DataType.EV_REPR] = [
            tensor.to(device=device, non_blocking=True) for tensor in data[DataType.EV_REPR]
        ]
    if DataType.OBJLABELS_SEQ in data:
        data[DataType.OBJLABELS_SEQ] = [
            labels.to(device=device) if labels is not None else None
            for labels in data[DataType.OBJLABELS_SEQ]
        ]
    if DataType.TOKEN_MASK in data:
        data[DataType.TOKEN_MASK] = [
            mask.to(device=device, non_blocking=True) for mask in data[DataType.TOKEN_MASK]
        ]

    return batch


def _iter_validation_batches(dataloader: Iterable, *, max_batches: Optional[int]) -> Iterable:
    if max_batches is None:
        yield from dataloader
        return
    for _, batch in zip(range(max_batches), dataloader):
        yield batch


@hydra.main(config_path=CONFIG_DIR, config_name="val", version_base="1.2")
def main(config: DictConfig) -> None:
    demo_cfg = _merge_demo_cfg(config)
    dynamically_modify_train_config(config)
    # Resolve config early to surface missing values.
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    dataset_name = config.dataset.name
    label_map = _resolve_label_map(dataset_name)

    gpu_index = config.hardware.gpus
    if torch.cuda.is_available() and isinstance(gpu_index, int) and gpu_index >= 0:
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    print(f"[demo] using device: {device}")

    print("[demo] loading data module...")
    data_module = fetch_data_module(config=config)
    data_module.setup(stage="validate")

    print("[demo] loading model from checkpoint...")
    module = fetch_model_module(config=config)
    module = type(module).load_from_checkpoint(str(Path(config.checkpoint)), full_config=config)
    module.eval().to(device)
    module.setup(stage="validate")

    wait_ms = int(demo_cfg.wait_ms)
    max_batches = demo_cfg.max_batches
    if max_batches is not None:
        max_batches = int(max_batches)
    headless_env = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    display = bool(demo_cfg.display)
    if headless_env and display:
        print("[demo] No display server detected; running in headless mode. "
              "Set demo.display=false to silence this message.")
        display = False
    show_ground_truth = bool(demo_cfg.show_ground_truth)
    output_path = demo_cfg.output_path
    fps = int(demo_cfg.fps)

    writer: Optional[cv2.VideoWriter] = None
    stacked_frame_size: Optional[tuple[int, int]] = None

    window_name = f"RVT {dataset_name} demo"
    if display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as err:
            print(f"[demo] Unable to open display window ({err}). Falling back to headless mode. "
                  f"Set demo.display=false to suppress this warning.")
            display = False

    val_loader = data_module.val_dataloader()
    batch_iter = _iter_validation_batches(val_loader, max_batches=max_batches)

    print("[demo] starting stream. Press 'q' in the display window to exit.")
    try:
        for batch_idx, batch in enumerate(batch_iter):
            _move_batch_to_device(batch, device=device)

            with torch.inference_mode():
                outputs = module.validation_step(batch=batch, batch_idx=batch_idx)

            if outputs is None or outputs.get(ObjDetOutput.SKIP_VIZ, False):
                continue

            ev_repr_tensor = outputs[ObjDetOutput.EV_REPR]
            if isinstance(ev_repr_tensor, torch.Tensor):
                ev_repr_np = ev_repr_tensor.detach().cpu().numpy()
            else:
                ev_repr_np = np.asarray(ev_repr_tensor)
            base_img = ev_repr_to_img(ev_repr_np)
            # base_img = VizCallbackBase.ev_repr_to_img(ev_repr_np)

            pred_boxes = outputs[ObjDetOutput.PRED_PROPH]
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.detach().cpu().numpy()

            pred_img = base_img.copy()
            draw_bboxes(pred_img, pred_boxes, labelmap=label_map)

            frames_to_stack = [pred_img]
            if show_ground_truth:
                label_boxes = outputs[ObjDetOutput.LABELS_PROPH]
                label_img = base_img.copy()
                draw_bboxes(label_img, label_boxes, labelmap=label_map)
                frames_to_stack.append(label_img)

            if len(frames_to_stack) == 1:
                stacked = frames_to_stack[0]
            else:
                stacked = np.vstack(frames_to_stack)

            if output_path and writer is None:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                height, width = stacked.shape[:2]
                stacked_frame_size = (width, height)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_file), fourcc, fps, stacked_frame_size)

            if writer is not None:
                if stacked_frame_size is None:
                    stacked_frame_size = (stacked.shape[1], stacked.shape[0])
                if (stacked.shape[1], stacked.shape[0]) != stacked_frame_size:
                    stacked = cv2.resize(stacked, stacked_frame_size)
                writer.write(stacked)

            if display:
                cv2.imshow(window_name, stacked)
                key = cv2.waitKey(wait_ms)
                if key & 0xFF == ord("q"):
                    print("[demo] received quit signal ('q').")
                    break
    finally:
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
