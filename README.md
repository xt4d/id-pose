# ID-Pose: Sparse-view Camera Pose Estimation by Inverting Diffusion Models 

<img src="docs/teaser.png" width="100%"/>

[<a href="https://arxiv.org/abs/2306.17140" target="_blank">Paper</a>] | [<a href="https://xt4d.github.io/id-pose-web/" target="_blank">Project Page</a>] | [<a href="https://xt4d.github.io/id-pose-web/#viewer" target="_blank">Interactive Examples</a>]

## TL;DR
- ID-Pose estimates camera poses of sparse input images (>= 2).
- ID-Pose inversely uses a view-conditioned diffusion model [Zero-1-to-3](https://zero123.cs.columbia.edu/) to find poses (no training required).
- ID-Pose generalizes to in-the-wild images as leveraging diffusion models pre-trained on large-scale data. 

## News
- [Feature] Initializing relative poses with estimated absolute elevations from input images. The estimation method and the source code are borrowed from [One-2-3-45](https://one-2-3-45.github.io/). This feature improves the metrics by about 3%-10% (tested on OmniObject3D). It also reduces the running time as elevations will not be probed. To use this feature, please download [indoor_ds_new.ckpt](https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp) to `ckpts/` and run the script with ```--est_elev```:
<pre>
python test_pose_estimation --input_json ./inputs/omni3d.json --exp_name omni3d <b>--est_elev</b>
</pre>
- Releasing the evaluation data & code. Please check the [Evaluation](#evaluation) section.
## Usage
### Installation
Create an environment with Python 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
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
Download a checkpoint from [Zero123 weights](https://huggingface.co/cvlab/zero123-weights/tree/main) to `ckpts/`.
```
mkdir -p ckpts/
wget -P ckpts/ https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt
```
You can also try the latest Zero123-XL checkpoint trained on the [Objaverse-XL](https://objaverse.allenai.org/objaverse-xl-paper.pdf) dataset of 10M+ objects!
```
wget -P ckpts/ https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```
Optional: To use the elevation feature, please download `indoor_ds_new.ckpt` from [LoFTR weights](https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp) and put it under `ckpts/`.
### Run demo
Requires around 28 GB of VRAM on an NVIDIA Tesla V100 GPU.
```
python test_pose_estimation.py --input_json ./inputs/omni3d.json --exp_name omni3d --ckpt_path ckpts/105000.ckpt
```
The results will be stored under `outputs/` with the name specified by ```--exp_name```.

Optional: To use the elevation feature, run the script with ```--est_elev```.

### Visualization
```
pip install jupyterlab
jupyter-lab viz.ipynb
```
### Evaluation
The evaluation data can be downloaded from [Google Drive](https://drive.google.com/file/d/1EU5D_enpxPTPaZHq9DDS2roninBULqEx/view?usp=sharing). Put the input json files under `inputs/` and the dataset folders under `data/`.

Run pose estimations on each dataset:
```
python test_pose_estimation.py --exp_name abo_tset --input_json inputs/abo_testset.json --bkg_threshold 0.9
python test_pose_estimation.py --exp_name omni3d_tset --input_json inputs/omni3d_testset.json --bkg_threshold 0.5
``` 
Run the evaluation script as:
```
python metric.py <outputs/exp_name/> <data/dataset_name/>
```

## Examples

<img src="docs/examples.png" width="100%"/>
The images outlined in <span style="color:red">red</span> are anchor views for which the camera poses have been manually found.

👉 Open <a href="https://xt4d.github.io/id-pose-web/#viewer" target="_blank">Interactive Viewer</a> to check more examples.

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
