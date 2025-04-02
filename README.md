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
EMF-GAN for [CUB](https://pan.baidu.com/s/1JfYS3binYGP9BLN_zUx-Jg).Password (5efd). Download and save it to ../saved_models/bird/<br />

* COCO<br />
Download the pre-trained text encoder for [COCO](https://drive.google.com/file/d/1e_AwWxbClxipEnasfz_QrhmLlv2-Vpyq/view?usp=sharing) and save it to ../coco/<br />
EMF-GAN for [COCO](https://pan.baidu.com/s/1j00282uqMA1u74dChMROiA).Password (9rbv). Download and save it to ../saved_models/coco/<br />

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

## Reference
- [DM-GAN: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1904.01310) [[code]](https://github.com/MinfengZhu/DM-GAN)
- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) [[code]](https://github.com/taoxugit/AttnGAN)
- [DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis](https://arxiv.org/abs/2008.05865) [[code]](https://github.com/tobran/DF-GAN.git)


