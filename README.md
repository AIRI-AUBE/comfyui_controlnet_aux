# ComfyUI's ControlNet Auxiliary Preprocessors
Plug-and-play [ComfyUI](https://github.com/comfyanonymous/ComfyUI) node sets for making [ControlNet](https://github.com/lllyasviel/ControlNet/) hint images

"anime style, a protest in the street, cyberpunk city, a woman with pink hair and golden eyes (looking at the viewer) is holding a sign with the text "ComfyUI ControlNet Aux" in bold, neon pink" on Flux.1 Dev

![](./examples/CNAuxBanner.jpg)

The code is copy-pasted from the respective folders in https://github.com/lllyasviel/ControlNet/tree/main/annotator. This fork is configured to load required model files locally (no runtime connection to the 🤗 Hub).

All credit & copyright goes to https://github.com/lllyasviel.

# Updates
Go to [Update page](./UPDATES.md) to follow updates

# Installation:
## Using ComfyUI Manager (recommended):
Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) and do steps introduced there to install this repo.

## Alternative:
If you're running on Linux, or non-admin account on windows you'll want to ensure `/ComfyUI/custom_nodes` and `comfyui_controlnet_aux` has write permissions.

There is now a **install.bat** you can run to install to portable if detected. Otherwise it will default to system and assume you followed ConfyUI's manual installation steps. 

If you can't run **install.bat** (e.g. you are a Linux user). Open the CMD/Shell and do the following:
  - Navigate to your `/ComfyUI/custom_nodes/` folder
  - Run `git clone https://github.com/Fannovel16/comfyui_controlnet_aux/`
  - Navigate to your `comfyui_controlnet_aux` folder
    - Portable/venv:
       - Run `path/to/ComfUI/python_embeded/python.exe -s -m pip install -r requirements.txt`
	- With system python
	   - Run `pip install -r requirements.txt`
  - Start ComfyUI

# Nodes
Please note that this repo only supports preprocessors making hint images (e.g. stickman, canny edge, etc).
All preprocessors except Inpaint are intergrated into `AIO Aux Preprocessor` node. 
This node allow you to quickly get the preprocessor but a preprocessor's own threshold parameters won't be able to set.
You need to use its node directly to set thresholds.

# Nodes (sections are categories in Comfy menu)
## Line Extractors
| Preprocessor Node           | sd-webui-controlnet/other |          ControlNet/T2I-Adapter           |
|-----------------------------|---------------------------|-------------------------------------------|
| Binary Lines                | binary                    | control_scribble                          |
| Canny Edge                  | canny                     | control_v11p_sd15_canny <br> control_canny <br> t2iadapter_canny |
| HED Soft-Edge Lines         | hed                       | control_v11p_sd15_softedge <br> control_hed |
| Standard Lineart            | standard_lineart          | control_v11p_sd15_lineart                 |
| Realistic Lineart           | lineart (or `lineart_coarse` if `coarse` is enabled) | control_v11p_sd15_lineart |
| Anime Lineart               | lineart_anime             | control_v11p_sd15s2_lineart_anime         |
| Manga Lineart               | lineart_anime_denoise     | control_v11p_sd15s2_lineart_anime         |
| M-LSD Lines                 | mlsd                      | control_v11p_sd15_mlsd <br> control_mlsd  |
| PiDiNet Soft-Edge Lines     | pidinet                   | control_v11p_sd15_softedge <br> control_scribble |
| Scribble Lines              | scribble                  | control_v11p_sd15_scribble <br> control_scribble |
| Scribble XDoG Lines         | scribble_xdog             | control_v11p_sd15_scribble <br> control_scribble |
| Fake Scribble Lines         | scribble_hed              | control_v11p_sd15_scribble <br> control_scribble |
| TEED Soft-Edge Lines        | teed                      | controlnet-sd-xl-1.0-softedge-dexined <br> control_v11p_sd15_softedge (Theoretically)
| Scribble PiDiNet Lines      | scribble_pidinet          | control_v11p_sd15_scribble <br> control_scribble |
| AnyLine Lineart             |                           | mistoLine_fp16.safetensors <br> mistoLine_rank256 <br> control_v11p_sd15s2_lineart_anime <br> control_v11p_sd15_lineart |

## Normal and Depth Estimators
| Preprocessor Node           | sd-webui-controlnet/other |          ControlNet/T2I-Adapter           |
|-----------------------------|---------------------------|-------------------------------------------|
| MiDaS Depth Map           | (normal) depth            | control_v11f1p_sd15_depth <br> control_depth <br> t2iadapter_depth |
| LeReS Depth Map           | depth_leres               | control_v11f1p_sd15_depth <br> control_depth <br> t2iadapter_depth |
| Zoe Depth Map             | depth_zoe                 | control_v11f1p_sd15_depth <br> control_depth <br> t2iadapter_depth |
| MiDaS Normal Map          | normal_map                | control_normal                            |
| BAE Normal Map            | normal_bae                | control_v11p_sd15_normalbae               |
| MeshGraphormer Hand Refiner ([HandRefinder](https://github.com/wenquanlu/HandRefiner))  | depth_hand_refiner | control_sd15_inpaint_depth_hand_fp16 |
| Depth Anything            |  depth_anything           | Depth-Anything |
| Zoe Depth Anything <br> (Basically Zoe but the encoder is replaced with DepthAnything)       | depth_anything | Depth-Anything |
| Normal DSINE              |                           | control_normal/control_v11p_sd15_normalbae |
| Metric3D Depth            |                           | control_v11f1p_sd15_depth <br> control_depth <br> t2iadapter_depth |
| Metric3D Normal           |                           | control_v11p_sd15_normalbae |
| Depth Anything V2         |                           | Depth-Anything |

## Faces and Poses Estimators
| Preprocessor Node           | sd-webui-controlnet/other |          ControlNet/T2I-Adapter           |
|-----------------------------|---------------------------|-------------------------------------------|
| DWPose Estimator                 | dw_openpose_full          | control_v11p_sd15_openpose <br> control_openpose <br> t2iadapter_openpose |
| OpenPose Estimator               | openpose (detect_body) <br> openpose_hand (detect_body + detect_hand) <br> openpose_faceonly (detect_face) <br> openpose_full (detect_hand + detect_body + detect_face)    | control_v11p_sd15_openpose <br> control_openpose <br> t2iadapter_openpose |
| MediaPipe Face Mesh         | mediapipe_face            | controlnet_sd21_laion_face_v2             | 
| Animal Estimator                 | animal_openpose           | control_sd15_animal_openpose_fp16 |

## Optical Flow Estimators
| Preprocessor Node           | sd-webui-controlnet/other |          ControlNet/T2I-Adapter           |
|-----------------------------|---------------------------|-------------------------------------------|
| Unimatch Optical Flow       |                           | [DragNUWA](https://github.com/ProjectNUWA/DragNUWA) |

### How to get OpenPose-format JSON?
#### User-side
This workflow will save images to ComfyUI's output folder (the same location as output images). If you haven't found `Save Pose Keypoints` node, update this extension
![](./examples/example_save_kps.png)

#### Dev-side
An array of [OpenPose-format JSON](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#json-output-format) corresponsding to each frame in an IMAGE batch can be gotten from DWPose and OpenPose using `app.nodeOutputs` on the UI or `/history` API endpoint. JSON output from AnimalPose uses a kinda similar format to OpenPose JSON:
```
[
    {
        "version": "ap10k",
        "animals": [
            [[x1, y1, 1], [x2, y2, 1],..., [x17, y17, 1]],
            [[x1, y1, 1], [x2, y2, 1],..., [x17, y17, 1]],
            ...
        ],
        "canvas_height": 512,
        "canvas_width": 768
    },
    ...
]
```

For extension developers (e.g. Openpose editor):
```js
const poseNodes = app.graph._nodes.filter(node => ["OpenposePreprocessor", "DWPreprocessor", "AnimalPosePreprocessor"].includes(node.type))
for (const poseNode of poseNodes) {
    const openposeResults = JSON.parse(app.nodeOutputs[poseNode.id].openpose_json[0])
    console.log(openposeResults) //An array containing Openpose JSON for each frame
}
```

For API users:
Javascript
```js
import fetch from "node-fetch" //Remember to add "type": "module" to "package.json"
async function main() {
    const promptId = '792c1905-ecfe-41f4-8114-83e6a4a09a9f' //Too lazy to POST /queue
    let history = await fetch(`http://127.0.0.1:8188/history/${promptId}`).then(re => re.json())
    history = history[promptId]
    const nodeOutputs = Object.values(history.outputs).filter(output => output.openpose_json)
    for (const nodeOutput of nodeOutputs) {
        const openposeResults = JSON.parse(nodeOutput.openpose_json[0])
        console.log(openposeResults) //An array containing Openpose JSON for each frame
    }
}
main()
```

Python
```py
import json, urllib.request

server_address = "127.0.0.1:8188"
prompt_id = '' #Too lazy to POST /queue

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

history = get_history(prompt_id)[prompt_id]
for o in history['outputs']:
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'openpose_json' in node_output:
            print(json.loads(node_output['openpose_json'][0])) #An list containing Openpose JSON for each frame
```
## Semantic Segmentation
| Preprocessor Node           | sd-webui-controlnet/other |          ControlNet/T2I-Adapter           |
|-----------------------------|---------------------------|-------------------------------------------|
| OneFormer ADE20K Segmentor  | oneformer_ade20k          | control_v11p_sd15_seg                     |
| OneFormer COCO Segmentor    | oneformer_coco            | control_v11p_sd15_seg                     |
| UniFormer Segmentor         | segmentation              |control_sd15_seg <br> control_v11p_sd15_seg|

## T2IAdapter-only
| Preprocessor Node           | sd-webui-controlnet/other |          ControlNet/T2I-Adapter           |
|-----------------------------|---------------------------|-------------------------------------------|
| Color Pallete               | color                     | t2iadapter_color                          |
| Content Shuffle             | shuffle                   | t2iadapter_style                          |

## Recolor
| Preprocessor Node           | sd-webui-controlnet/other |          ControlNet/T2I-Adapter           |
|-----------------------------|---------------------------|-------------------------------------------|
| Image Luminance             | recolor_luminance         | ioclab_sd15_recolor <br> sai_xl_recolor_256lora <br> bdsqlsz_controlllite_xl_recolor_luminance |
| Image Intensity             | recolor_intensity         | Idk. Maybe same as above? |

# Examples
> A picture is worth a thousand words

![](./examples/ExecuteAll1.jpg)
![](./examples/ExecuteAll2.jpg)

# Testing workflow
https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/examples/ExecuteAll.png
Input image: https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/examples/comfyui-controlnet-aux-logo.png

# Q&A:
## Why some nodes doesn't appear after I installed this repo?

This repo has a new mechanism which will skip any custom node can't be imported. If you meet this case, please create a issue on [Issues tab](https://github.com/Fannovel16/comfyui_controlnet_aux/issues) with the log from the command line.

## DWPose/AnimalPose only uses CPU so it's so slow. How can I make it use GPU?
There are two ways to speed-up DWPose: using TorchScript checkpoints (.torchscript.pt) checkpoints or ONNXRuntime (.onnx). TorchScript way is little bit slower than ONNXRuntime but doesn't require any additional library and still way way faster than CPU. 

A torchscript bbox detector is compatiable with an onnx pose estimator and vice versa.
### TorchScript
Set `bbox_detector` and `pose_estimator` according to this picture. You can try other bbox detector endings with `.torchscript.pt` to reduce bbox detection time if input images are ideal.
![](./examples/example_torchscript.png)
### ONNXRuntime
If onnxruntime is installed successfully and the checkpoint used endings with `.onnx`, it will replace default cv2 backend to take advantage of GPU. Note that if you are using NVidia card, this method currently can only works on CUDA 11.8 (ComfyUI_windows_portable_nvidia_cu118_or_cpu.7z) unless you compile onnxruntime yourself.

1. Know your onnxruntime build:
* * NVidia CUDA 11.x or bellow/AMD GPU: `onnxruntime-gpu`
* * NVidia CUDA 12.x: `onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`
* * DirectML: `onnxruntime-directml`
* * OpenVINO: `onnxruntime-openvino`

Note that if this is your first time using ComfyUI, please test if it can run on your device before doing next steps.

2. Add it into `requirements.txt`

3. Run `install.bat` or pip command mentioned in Installation

![](./examples/example_onnx.png)

# Offline preprocessor assets
Runtime downloads are disabled. Place model assets under `ckpts/` in this repository, or set `AUX_ANNOTATOR_CKPTS_PATH` and use the same relative layout there.

To download the runtime asset set automatically, run:

```bash
python download_models.py
```

If `huggingface_hub` is not installed in that environment, install it first:

```bash
python -m pip install huggingface_hub
```

* anime_face_segment: `ckpts/bdsqlsz/qinglong_controlnet-lllite/Annotators/UNet.pth`, `ckpts/skytnt/anime-seg/isnetis.ckpt`
* densepose: `ckpts/LayerNorm/DensePose-TorchScript-with-hint-image/densepose_r50_fpn_dl.torchscript`
* depth_anything: local transformers repo under `ckpts/LiheYoung/depth-anything-large-hf`, `ckpts/LiheYoung/depth-anything-base-hf`, or `ckpts/LiheYoung/depth-anything-small-hf` with `config.json`, `preprocessor_config.json`, and `model.safetensors` or `pytorch_model.bin`
* depth_anything_v2: `ckpts/depth-anything/Depth-Anything-V2-Small/depth_anything_v2_vits.pth`, `ckpts/depth-anything/Depth-Anything-V2-Base/depth_anything_v2_vitb.pth`, `ckpts/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth`, `ckpts/depth-anything/Depth-Anything-V2-Giant/depth_anything_v2_vitg.pth`, `ckpts/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/depth_anything_v2_metric_vkitti_vitl.pth`, `ckpts/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/depth_anything_v2_metric_hypersim_vitl.pth`
* diffusion_edge: `ckpts/hr16/Diffusion-Edge/diffusion_edge_indoor.pt`, `ckpts/hr16/Diffusion-Edge/diffusion_edge_urban.pt`, or `ckpts/hr16/Diffusion-Edge/diffusion_edge_natrual.pt`
* dsine: `ckpts/hr16/Diffusion-Edge/dsine.pt`
* dwpose: bbox detector at `ckpts/yzd-v/DWPose/yolox_l.onnx`, `ckpts/hr16/yolox-onnx/yolox_l.torchscript.pt`, `ckpts/hr16/yolo-nas-fp16/yolo_nas_l_fp16.onnx`, `ckpts/hr16/yolo-nas-fp16/yolo_nas_m_fp16.onnx`, or `ckpts/hr16/yolo-nas-fp16/yolo_nas_s_fp16.onnx`; pose estimator at `ckpts/yzd-v/DWPose/dw-ll_ucoco_384.onnx` or `ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt`
* animal_pose: bbox detector at `ckpts/yzd-v/DWPose/yolox_l.onnx`, `ckpts/hr16/yolox-onnx/yolox_l.torchscript.pt`, `ckpts/hr16/yolo-nas-fp16/yolo_nas_l_fp16.onnx`, `ckpts/hr16/yolo-nas-fp16/yolo_nas_m_fp16.onnx`, or `ckpts/hr16/yolo-nas-fp16/yolo_nas_s_fp16.onnx`; pose estimator at `ckpts/hr16/UnJIT-DWPose/rtmpose-m_ap10k_256.onnx` or `ckpts/hr16/DWPose-TorchScript-BatchSize5/rtmpose-m_ap10k_256_bs5.torchscript.pt`
* hed: `ckpts/lllyasviel/Annotators/ControlNetHED.pth`
* leres: `ckpts/lllyasviel/Annotators/res101.pth`, `ckpts/lllyasviel/Annotators/latest_net_G.pth`
* lineart: `ckpts/lllyasviel/Annotators/sk_model.pth`, `ckpts/lllyasviel/Annotators/sk_model2.pth`
* lineart_anime: `ckpts/lllyasviel/Annotators/netG.pth`
* manga_line: `ckpts/lllyasviel/Annotators/erika.pth`
* mesh_graphormer: `ckpts/hr16/ControlNet-HandRefiner-pruned/graphormer_hand_state_dict.bin`, `ckpts/hr16/ControlNet-HandRefiner-pruned/hrnetv2_w64_imagenet_pretrained.pth`; Graphormer config is bundled at `src/custom_mesh_graphormer/modeling/bert/bert-base-uncased`
* metric3d: `ckpts/JUGGHM/Metric3D/metric_depth_vit_small_800k.pth`, `ckpts/JUGGHM/Metric3D/metric_depth_vit_large_800k.pth`, or `ckpts/JUGGHM/Metric3D/metric_depth_vit_giant2_800k.pth`
* midas: local transformers repo under `ckpts/Intel/dpt-hybrid-midas` or `ckpts/Intel/dpt-large` with `config.json`, `preprocessor_config.json`, and `model.safetensors` or `pytorch_model.bin`
* mlsd: `ckpts/lllyasviel/Annotators/mlsd_large_512_fp32.pth`
* normalbae: `ckpts/lllyasviel/Annotators/scannet.pt`
* oneformer: local transformers repo under `ckpts/shi-labs/oneformer_ade20k_swin_large` or `ckpts/shi-labs/oneformer_coco_swin_large` with `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`, and `model.safetensors` or `pytorch_model.bin`
* open_pose: `ckpts/lllyasviel/Annotators/body_pose_model.pth`, `ckpts/lllyasviel/Annotators/hand_pose_model.pth`, `ckpts/lllyasviel/Annotators/facenet.pth`
* pidi: `ckpts/lllyasviel/Annotators/table5_pidinet.pth`
* sam: local transformers repo under `ckpts/facebook/sam-vit-base`, `ckpts/facebook/sam-vit-large`, or `ckpts/facebook/sam-vit-huge` with `config.json`, `preprocessor_config.json`, and `model.safetensors` or `pytorch_model.bin`
* teed: `ckpts/bdsqlsz/qinglong_controlnet-lllite/Annotators/7_model.pth`
* uniformer: `ckpts/lllyasviel/Annotators/upernet_global_small.pth`
* unimatch: `ckpts/hr16/Unimatch/gmflow-scale2-regrefine6-mixdata.pth`, `ckpts/hr16/Unimatch/gmflow-scale2-mixdata.pth`, or `ckpts/hr16/Unimatch/gmflow-scale1-mixdata.pth`
* zoe: local transformers repo under `ckpts/Intel/zoedepth-nyu-kitti` with `config.json`, `preprocessor_config.json`, and `model.safetensors` or `pytorch_model.bin`
* zoe_depth_anything: local transformers repo under `ckpts/Intel/zoedepth-nyu-kitti` with `config.json`, `preprocessor_config.json`, and `model.safetensors` or `pytorch_model.bin`
* torch_backbones: `ckpts/torch/mobilenet_v2-b0353104.pth`, `ckpts/torch/vgg16-397923af.pth`, `ckpts/torch/resnet101-cd907fc2.pth`, `ckpts/torch/efficientnet_b7_lukemelas-dcc49843.pth`, `ckpts/torch/swin_b-68c6b09e.pth`
# 2000 Stars 😄
<a href="https://star-history.com/#Fannovel16/comfyui_controlnet_aux&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Fannovel16/comfyui_controlnet_aux&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Fannovel16/comfyui_controlnet_aux&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Fannovel16/comfyui_controlnet_aux&type=Date" />
  </picture>
</a>

Thanks for yalls supports. I never thought the graph for stars would be linear lol.
