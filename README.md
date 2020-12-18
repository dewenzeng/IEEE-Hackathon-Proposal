# How to use our code

1. Download the data from this [link](https://drive.google.com/drive/folders/1g2YDMp4wCALeQLBXi-NsY6eW_wExfHO-?usp=sharing) and the pretrained model from this [link](https://drive.google.com/drive/folders/1ASbqSiKx7d1m1nvSW6h0dba_HlIXA_oq?usp=sharing).
2. run the predition command and you will get segmentation result in ./result folder
```
python prediction.py --model_infection_dir YOUR_MODEL_INFECTION_DIR --model_lung_dir YOUR_MODEL_LUNG_DIR --test_data_dir YOUR_TEST_DATA_DIR
```
