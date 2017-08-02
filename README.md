# An Improved Deep Learning Architecture for Person Re-Identification

[paper link](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf)

## Environment
* Ubuntu 16.04
* Python 2.7.12
* TensorFlow 1.2.1
* CUDA 8.0
* cudnn 5.1
* NVIDIA GTX Titan X (Pascal)

## Installation
```
git clone https://github.com/thomaspark-pkj/person-reid.git
```

## Prepare dataset
Download CUHK03 dataset from [CUHK Person Re-identification Datasets](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) then extract file.
```
python cuhk03_dataset.py your_dataset_path
```

## Traininig
```
python run.py --data_dir=your_dataset_path
```

## Validation
```
python run.py --mode=val --data_dir=your_dataset_path
```
##### Result
Accuracy can be different slightly because image pair is generated randomly in validation dataset.
```
Accuracy: 0.846667
```

## Test
```
python run.py --mode=test --image1=your_dataset_path/labeled/val/0000_00.jpg --image2=your_dataset_path/labeled/val/0000_05.jpg
```
##### Result
```
True
```