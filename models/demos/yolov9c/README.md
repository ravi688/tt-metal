# Yolov9c Demo
Demo showcasing Yolov9c running on Wormhole - n150, n300 using ttnn.

## Platforms:
    WH N150, N300

## Introduction:
Yolov9 marks a significant advancement in real-time object detection, introducing groundbreaking techniques such as Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). Yolov9c is a compact and optimized variant of Yolov9, designed for efficient object detection with reduced computational overhead. It balances speed and accuracy.

## Details:
The entry point to functional_yolov9c model is YoloV9 in `models/demos/yolov9c/tt/ttnn_yolov9c.py`. The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset) under YOLOv9c.

## How to Run:
If running on Wormhole N300 (not required for N150 or Blackhole), the following environment variable needs to be set as the model requires at least 8x8 core grid size:
```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```
#### Note:
- Use `yolov9c-seg.pt` pre-trained weights for segmentation tasks and `yolov9c.pt` pre-trained weights for detection in Tests and Demos.
- Set the `enable_segment` flag accordingly when initializing the TTNN model in tests and demos. Segmentation task is set as default in model.

Use the following command to run the Yolov9c model :
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov9c/test_ttnn_yolov9c.py::test_yolov9c
```
### Demo
Use the following command to run the demo for Instance Segmentation task with pre-trained weights:
```
pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo[segment-True-tt_model-models/demos/yolov9c/demo/image.png-device_params0]
```
Use the following command to run the demo for Object Detection task with pre-trained weights:
```
pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo[detect-True-tt_model-models/demos/yolov9c/demo/image.png-device_params0]
```

#### Outputs
- The Demo outputs are saved inside this directory: `models/demos/yolov9c/demo/runs`

### Model performant running

#### For 640x640:
- end-2-end perf with Trace+2CQ for Segmentation is 54 FPS

  ```bash
  pytest models/demos/yolov9c/tests/perf/test_e2e_performant.py::test_e2e_performant[resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```

- device perf for Segementation is 52 FPS

  ```bash
  pytest models/demos/yolov9c/tests/perf/test_perf.py::test_perf_device_bare_metal_yolov9c
  ```
