# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov9c.demo.demo_seg_utils import postprocess, preprocess
from models.demos.yolov9c.demo.demo_utils import load_coco_class_names
from models.demos.yolov9c.reference import yolov9c
from models.demos.yolov9c.tt import ttnn_yolov9c
from models.demos.yolov9c.tt.model_preprocessing import create_yolov9c_input_tensors, create_yolov9c_model_parameters
from models.experimental.yolo_evaluation.yolo_common_evaluation import save_yolo_predictions_by_model
from models.experimental.yolo_evaluation.yolo_evaluation_utils import LoadImages
from models.experimental.yolo_evaluation.yolo_evaluation_utils import postprocess as obj_postprocess
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


def get_consistent_color(index):
    cmap = plt.get_cmap("tab20")
    color = cmap(index % 20)[:3]
    return tuple(int(c * 255) for c in color)


def save_seg_predictions_by_model(result, save_dir, image_path, model_name):
    os.makedirs(os.path.join(save_dir, model_name), exist_ok=True)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = result.masks.data.cpu().detach().numpy()
    mask_h, mask_w = masks.shape[1], masks.shape[2]

    image = cv2.resize(image, (mask_w, mask_h))
    overlay = image.copy()

    for i in range(len(masks)):
        mask = masks[i]
        color = get_consistent_color(i)
        mask_rgb = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            mask_rgb[:, :, c] = (mask * color[c]).astype(np.uint8)

        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * mask_rgb[mask_bool]).astype(np.uint8)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(save_dir, model_name, f"segmentation_{timestamp}.jpg")
    cv2.imwrite(out_path, overlay_bgr)
    logger.info(f"Saved to {out_path}")


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source",
    [
        "models/demos/yolov9c/demo/image.png",
        # "models/sample_data/huggingface_cat_image.jpg", # Uncomment to run the demo with another image.
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",  # To run the demo for instance segmentation
        # "detect",  # Uncomment to run the demo for Object Detection
    ],
)
def test_demo(device, source, model_type, use_weights_from_ultralytics, model_task, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()

    weights = "yolov9c-seg.pt" if model_task == "segment" else "yolov9c.pt"
    enable_segment = model_task == "segment"

    state_dict = None

    if model_type == "torch_model":
        if use_weights_from_ultralytics:
            torch_model = YOLO(weights)
            torch_model.eval()
            state_dict = torch_model.state_dict()
        model = yolov9c.YoloV9(enable_segment=enable_segment)
        state_dict = model.state_dict() if state_dict is None else state_dict

        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2

        model.load_state_dict(new_state_dict)
        # model = torch_model
        model.eval()

        logger.info("Inferencing [Torch] Model")
    else:
        torch_input, ttnn_input = create_yolov9c_input_tensors(device)
        if use_weights_from_ultralytics:
            torch_model = YOLO(weights)
            torch_model.eval()
            state_dict = torch_model.state_dict()

        model = yolov9c.YoloV9(enable_segment=enable_segment)
        state_dict = model.state_dict() if state_dict is None else state_dict

        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2

        model.load_state_dict(new_state_dict)
        model.eval()
        parameters = create_yolov9c_model_parameters(model, torch_input, device=device)
        model = ttnn_yolov9c.YoloV9(device, parameters, enable_segment=enable_segment)
        logger.info("Inferencing [TTNN] Model")

    save_dir = "models/demos/yolov9c/demo/runs"
    dataset = LoadImages(path=source)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, res=(640, 640))

        if model_type == "torch_model":
            preds = model(im)
            if enable_segment:
                results = postprocess(preds, im, im0s, batch)
                save_seg_predictions_by_model(results[0], save_dir, source, model_type)
            else:
                results = obj_postprocess(preds, im, im0s, batch, names)
                save_yolo_predictions_by_model(results[0], save_dir, source, model_type)

        else:
            img = torch.permute(im, (0, 2, 3, 1))
            ttnn_im = ttnn.from_torch(img, dtype=ttnn.bfloat16)

            preds = model(ttnn_im)
            preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)
            if enable_segment:
                detect1_out, detect2_out, detect3_out = [
                    ttnn.to_torch(tensor, dtype=torch.float32) for tensor in preds[1][0]
                ]

                mask = ttnn.to_torch(preds[1][1], dtype=torch.float32)
                proto = ttnn.to_torch(preds[1][2], dtype=torch.float32)
                proto = proto.reshape((1, 160, 160, 32)).permute((0, 3, 1, 2))

                preds[1] = [[detect1_out, detect2_out, detect3_out], mask, proto]

                results = postprocess(preds, im, im0s, batch)
                for i in range(len(results)):
                    save_seg_predictions_by_model(results[i], save_dir, paths[i], model_type)
            else:
                results = obj_postprocess(preds[0], im, im0s, batch, names)[0]
                save_yolo_predictions_by_model(results, save_dir, source, model_type)

    logger.info("Inference done")
