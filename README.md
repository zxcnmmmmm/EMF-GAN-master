  # EMF-GAN-master

## Main Requirements
- python 3.8
- torch 

## Installation
```
git clone https://github.com/zxcnmmmmm/EMF-GAN-master.git
cd EMF-GAN-master
pip install -r requirements.txt
```

## Prepare data
Preprocessed metadata for [CUB](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and save them to data/.<br />
Image data for [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) [coco2014](http://cocodataset.org/#download). Extract them to data/.<br />

## Pretrained Model
* CUB<br />
Download the pre-trained text encoder for [CUB](https://drive.google.com/file/d/1rzfcCvGwU8vLCrn5reWxmrAMms6WQGA6/view?usp=sharing) and save it to ../bird/<br />
EMF-GAN for [CUB](https://pan.baidu.com/s/1Y7f1zhIGoSo_rit_UlTG5A).Password (1008). Download and save it to ../saved_models/bird/<br />

* COCO<br />
Download the pre-trained text encoder for [COCO](https://drive.google.com/file/d/1e_AwWxbClxipEnasfz_QrhmLlv2-Vpyq/view?usp=sharing) and save it to ../coco/<br />
EMF-GAN for [COCO](https://pan.baidu.com/s/1Y7f1zhIGoSo_rit_UlTG5A).Password (1008). Download and save it to ../saved_models/coco/<br />

## Synthesize images
  ```
  cd EMF-GAN/code/
  ```
- CUB: `bash src/test.sh ./cfg/bird.yml`
- COCO: `bash src/test.sh ./cfg/coco.yml`

## Train model
  ```
  cd EMF-GAN/code/
  ```
- CUB: `bash src/train.sh ./cfg/bird.yml`
- COCO: `bash src/train.sh ./cfg/coco.yml`


## Results
Example results on the CUB and COCO.
<div align="left;">
  <img src="https://github.com/zxcnmmmmm/EMF-GAN-master/blob/main/example.jpg" width="400">
</div>


