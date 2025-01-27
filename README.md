# SF-MVSNet


🔧# Installation
```
conda create -n gomvs python=3.8 # use python 3.8
conda activate gomvs
pip install -r requirements.txt
#Our code is trained and tested on NVIDIA RTX 3090 GPU (with python=3.8, torch=1.12.1 cuda=11.3)
```
If you are using NVIDIA RTX 3090 GPU too, please use the following commands reinstall pytorch
```
# uninstall
pip uninstall pytorch

# install
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# or
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

# Dataset
We use the dataset [GoMVS](https://github.com/Wuuu3511/GoMVS) provided, which is the same for many other MVSNet methods except normals. Please refer to [GoMVS](https://github.com/Wuuu3511/GoMVS) for download link.

# Run
*Note SF-MVSNet require roughly 20GB+ GPU memory for 1 batchsize, and We didn't test `finetune.py` because our goal is to test TNT dataset without fine-tuning.*

## Train
Please specify MVS_TRAINING and NORMAL_PATH in scripts/train/*.sh, then run the following bash to train on DTU dataset
```
bash scripts/train/train_dtu.sh
```

## Evaluation
### Pretrained model
Pretrained model can be downloaded in this [link](https://pan.baidu.com/s/1M11V5CKIeWt11xb4P_qLFA), verify code is *t52u*.

### DTU Dataset
For quantitative results:
1. build *fusibile* from [this repo](https://github.com/YoYo000/fusibile). Instructions are in [MVSNet](https://github.com/YoYo000/MVSNet) Post-Processing section.
2. run `bash scripts/test/test_dtu.sh` to get depth map and point clouds.
3. run DTU matlab code.


### TNT Dataset
Run the following bash to get depth map and point clouds.
```
# get depth map
bash scripts/test/test_tnt.sh

# get point clouds
bash scripts/test/dynamic_fusion.sh
```


# Acknowledgments
We borrow the code from [TransMVSNet](https://github.com/megvii-research/TransMVSNet),[GoMVS](https://github.com/Wuuu3511/GoMVS), We express gratitude for their marvelous contributions!
