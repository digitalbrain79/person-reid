# Tensorflow implementation of "An Improved Deep Learning Architecture for Person Re-Identification"

[paper link](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf)

## Prepare dataset
Download CUHK03 dataset from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html
```
python cuhk03_dataset.py your_dataset_path
```

## Train
```
python run.py --data_dir=your_dataset_path
```

## Validation
```
python run.py --mode=val --data_dir=your_dataset_path
```

## Result
