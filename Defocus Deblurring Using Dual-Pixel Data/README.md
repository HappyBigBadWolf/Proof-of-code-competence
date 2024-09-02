
# README

Pytorch version of Defocus Deblurring Using Dual-Pixel Data.

Pre-trained [model](https://mailsdueducn-my.sharepoint.com/:u:/g/personal/201700181055_mail_sdu_edu_cn/EZfA2nONQwpGniF02GEjxOABDCLw1jwqaZxnJogbw-xkVw?e=TDLWlS).

## Dataset

First, please follow the same steps of repository [defocus-deblurring-dual-pixel](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) to crop images into patches.

Then, re-orginaze patches:

```
DualPixelCanon:
    - train:
        - source:
            00000.png
            00001.png
            ...
        - target:
            00000.png
            00001.png
            ...
        - l_view:
            00000.png
            00001.png
            ...
        - r_view:
            00000.png
            00001.png
            ...
    - valid:
        ...
    - test:
        ...
```

## Setup

create your conda env, i.e., `conda create -n dddp python=3.8`.

install pytorch>=1.7.0. i.e., `conda install pytorch torchvision -c pytorch`

`cd path_to_dddp` (root of this project), i.e., `cd /home/user/models/dddp`

install dependencies: `pip install -r requirements.txt`

change `cfg.DATA.DIR` in `root_of_dddp/src/configs/configs.py` to path to your dataset.

run following code in terminal to start.

`python src/main.py [IDName] [Dataset] [BatchSize] [Resume] [Train] [Validation] [Test] [GPU]`

i.e. `python src/main.py MyID DualPixelCanon 5 false true true true [0]`