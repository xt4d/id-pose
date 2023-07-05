# ID-Pose: Sparse-view Camera Pose Estimation by Inverting Diffusion Models 

<img src="docs/teaser.png" width="100%"/>

[[Paper](https://arxiv.org/abs/2306.17140)] | [[Project Page](https://xt4d.github.io/id-pose-web/)]

## TL;DR
- ID-Pose estimates camera poses of input images (>= 2).
- ID-Pose inversely uses a view-conditioned diffusion model ([Zero-1-to-3](https://zero123.cs.columbia.edu/)) to find poses. 
- ID-Pose generalizes to in-the-wild images as leveraging diffusion models pre-trained on large-scale data. 


## Usage
### Installation
Create an environment with Python 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
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
Download the [Zero-1-to-3 checkpoint](https://huggingface.co/cvlab/zero123-weights/tree/main) to `ckpts/`
```
mkdir -p ckpts/
wget -P ckpts/ https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt
```

### Run demo
Requires around 28 GB of VRAM on an NVIDIA Tesla V100 GPU.
```
python test_pose_estimation.py --input_json ./inputs/omni3d.json --exp_name omni3d
```
The results will be stored under `outputs/` with the name of --exp_name.

### Visualization
```
pip install jupyterlab
jupyter-lab viz.ipynb
```


## Work in progress
- 3D reconstruction with posed images.
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
