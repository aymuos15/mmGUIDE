## Installation: [[Reference]](https://mmdetection.readthedocs.io/en/latest/get_started.html)

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch #! Do not know how long this will last as Torch does not support conda anymore.
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

<ins><b>Warning</b></ins>: This can be extremely tricky and I honestly got lucky. Try debugging and matching versions as you go if you face errors. Pasting errors to an LLM is a nice hacky way to mostly get the right versioning. 

### To verify [[Reference]](https://mmdetection.readthedocs.io/en/latest/get_started.html)

```
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```