# ID-Pose: Sparse-view Camera Pose Estimation by Inverting Diffusion Models 

<img src="docs/teaser.png" width="100%"/>

[<a href="https://arxiv.org/abs/2306.17140" target="_blank">Paper</a>] | [<a href="https://xt4d.github.io/id-pose-web/" target="_blank">Project Page</a>] | [<a href="https://huggingface.co/spaces/tokenid/ID-Pose" target="_blank">HF Demo</a>] | [<a href="https://xt4d.github.io/id-pose-web/viewer.html" target="_blank">Examples</a>]

## TL;DR
- ID-Pose estimates camera poses of sparse-view images of a 3D object (appearance overlaps not required).
- ID-Pose inversely uses the off-the-shelf [Zero-1-to-3](https://zero123.cs.columbia.edu/) to estimate camera poses by iteratively minimizing denoising errors given input images.
- ID-Pose is a zero-shot method that requires NO additional model training or finetuning.
- ID-Pose exhibits strong generalization ability on open-world images as the method effectively leverages the image priors from Zero123 (StableDiffusion).

## News
- [2023-11-12] We incoporate "absolute elevation estimation" as the default setting. We update the default values of the following parameters: ```--probe_min_timestep```, ```--probe_max_timestep```, ```--min_timestep```, ```--max_timestep```. 
- [2023-09-11] We introduce a new feature that initializing relative poses with estimated absolute elevations from input images. The estimation method and the source code are borrowed from [One-2-3-45](https://one-2-3-45.github.io/). This feature improves the metrics by about 3%-10% (tested on OmniObject3D). It also reduces the running time as elevations will not be probed. 
- [2023-09-11] We release the evaluation data & code. Please check the [Evaluation](#evaluation) section.
## Usage
### Installation
Create an environment with Python 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
```
git clone https://github.com/xt4d/id-pose.git
cd id-pose/
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
```

### Download checkpoints
1. Download `zero123-xl.ckpt` to `ckpts/`.
```
mkdir -p ckpts/
wget -P ckpts/ https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```
2. Download `indoor_ds_new.ckpt` from [LoFTR weights](https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp) to `ckpts/`.
### Run examples
Running requires around 28 GB of VRAM on an NVIDIA Tesla V100 GPU.
```
## Example 1: Image folder ##
python test_pose_estimation.py --input ./data/demo/lion/ --output outputs/demo/

## Example 2: Structured evaluation data ##
## Include --no_rembg if the images do not have a background.
python test_pose_estimation.py --input ./inputs/real.json --output outputs/real --no_rembg

## Example 3: Structured evaluation data ##
python test_pose_estimation.py --input ./inputs/omni3d.json --output outputs/omni3d --no_rembg
```
The results will be stored under the directory specified by `--output`.

### Visualization
```
pip install jupyterlab
jupyter-lab viz.ipynb
```

## Use your own data
Step 1: Create an image folder. For example:
```
mkdir -p data/demo/lion/
```

Step 2: Put the images under the folder. For example:
```
lion
├── 000.jpg
├── 001.jpg
```

Step 3: Run estimation:
```
python test_pose_estimation.py --input ./data/demo/lion/ --output outputs/demo/
```
The results will be stored under `outputs/demo/`.

## Evaluation
The evaluation data can be downloaded from [Google Drive](https://drive.google.com/file/d/1EU5D_enpxPTPaZHq9DDS2roninBULqEx/view?usp=sharing). Put the input json files under `inputs/` and the dataset folders under `data/`.

Run pose estimations on each dataset:
```
python test_pose_estimation.py --input inputs/abo_testset.json --output outputs/abo_tset --no_rembg --bkg_threshold 0.9
python test_pose_estimation.py --input inputs/omni3d_testset.json --output outputs/omni3d_tset --no_rembg --bkg_threshold 0.5
``` 
Run the evaluation script as:
```
python metric.py --input outputs/abo_tset --gt data/abo/
python metric.py --input outputs/omni3d_tset --gt data/omni3d/
```

## Examples

<img src="docs/examples.png" width="100%"/>
The images outlined in <span style="color:red">red</span> are anchor views for which the camera poses have been manually found.

👉 Open <a href="https://xt4d.github.io/id-pose-web/viewer.html" target="_blank">Interactive Viewer</a> to check more examples.

## Work in progress
- 3D reconstruction with posed images.
- Reduce the running time of ID-Pose.
- Upgrade ID-Pose to estimate 6DOF poses.


##  Citation
```
@article{cheng2023id,
  title={ID-Pose: Sparse-view Camera Pose Estimation by Inverting Diffusion Models},
  author={Cheng, Weihao and Cao, Yan-Pei and Shan, Ying},
  journal={arXiv preprint arXiv:2306.17140},
  year={2023}
}
```
