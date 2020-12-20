# How to use our code

1. Download the data from this [link](https://drive.google.com/drive/folders/1g2YDMp4wCALeQLBXi-NsY6eW_wExfHO-?usp=sharing) and the pretrained model from this [link](https://drive.google.com/drive/folders/1ASbqSiKx7d1m1nvSW6h0dba_HlIXA_oq?usp=sharing).
2. run the predition command and you will get segmentation result in ./result folder
```
python prediction.py --model_infection_dir YOUR_MODEL_INFECTION_DIR --model_lung_dir YOUR_MODEL_LUNG_DIR --test_data_dir YOUR_TEST_DATA_DIR
```

## Input and Output

- **Input**, we use the 3D CT volume (512x512x512) as the input of our system.
- **Output**, the output of the system is a 3D binary segmentation mask of the corresponding CT image. Regions with pixel value 1 mean these areas are infected with COVID-19, Regions with pixel value 0 are normal tissues.
  
An example of the segmentation result can be seen in the figure below. The left figure shows the original slice (extracted from a 3D CT volume), the right figure shows the segmentation result. Red means the region is infected.

![example](/images/example.png)

## Performance Evaluation

Our work is based on the pretrained model from this paper: **A Rapid, Accurate and Machine-Agnostic Segmentation and Quantification Method for CT-Based COVID-19 Diagnosis** [link](https://ieeexplore.ieee.org/abstract/document/9115057). We modified the parameters of the original model and add quantization to them in order to reduce the computation overhead. We also add security protocol on top of the network architecture. The security protocols we use (e.g., PAHE and GC) are standard protocols, which are not included in our code. You can refer to [PAHE](https://github.com/snucrypto/HEAAN) and [GC](https://github.com/ojroques/garbled-circuit) if you are interested in their detailed implementation. 

#### What to evaluate?

Since the ground truth label of the CT volume is not provided by the original paper, we use their pretrained model to generate segmentation masks on the testset and use these masks as the ground truth baseline of our method. We need to evaluate **how much the accuracy drops** once we apply the security protocol in the inference phase. Therefore, we report the segmentation accuracy as Dice coefficient in our prediction.py script. The Dice coefficient of our baseline (no quantization and security protocol) is 1.0.


