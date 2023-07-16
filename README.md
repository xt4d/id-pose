# ID-Pose: Sparse-view Camera Pose Estimation by Inverting Diffusion Models 

<img src="docs/teaser.png" width="100%"/>

[<a href="https://arxiv.org/abs/2306.17140" target="_blank">Paper</a>] | [<a href="https://xt4d.github.io/id-pose-web/" target="_blank">Project Page</a>] | [<a href="https://xt4d.github.io/id-pose-web/#viewer" target="_blank">Interactive Demo</a>]

## TL;DR
- ID-Pose estimates camera poses of sparse input images (>= 2).
- ID-Pose inversely uses a view-conditioned diffusion model (<a href="https://zero123.cs.columbia.edu/" target="_blank">Zero-1-to-3</a>) to find poses (no training required).
- ID-Pose generalizes to in-the-wild images as leveraging diffusion models pre-trained on large-scale data. 

## Usage
### Installation
Create an environment with Python 3.9 (Recommend to use <a href="https://www.anaconda.com/download/" target="_blank">Anaconda</a> or <a href="https://docs.conda.io/en/latest/miniconda.html" target="_blank">Miniconda</a>)
```
git clone https://github.com/xt4d/id-pose.git
cd id-pose/
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

### Download checkpoint
Download the <a href="https://huggingface.co/cvlab/zero123-weights/tree/main" target="_blank">Zero123 checkpoint</a> to `ckpts/`
```
mkdir -p ckpts/
wget -P ckpts/ https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt
```
You can also try the latest Zero123-XL checkpoint trained on the <a href="https://objaverse.allenai.org/objaverse-xl-paper.pdf" target="_blank">Objaverse-XL</a> dataset of 10M+ objects!
```
wget -P ckpts/ https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```

### Run demo
Requires around 28 GB of VRAM on an NVIDIA Tesla V100 GPU.
```
python test_pose_estimation.py --input_json ./inputs/omni3d.json --exp_name omni3d --ckpt_path ckpts/105000.ckpt
```
The results will be stored under `outputs/` with the name of --exp_name.

### Visualization
```
pip install jupyterlab
jupyter-lab viz.ipynb
```

## Examples

<img src="docs/examples.png" width="100%"/>
The images outlined in <span style="color:red">red</span> are anchor views for which the camera poses have been manually found.

👉 Open <a href="https://xt4d.github.io/id-pose-web/#viewer" target="_blank">Interactive Demo</a> to check more examples.

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
