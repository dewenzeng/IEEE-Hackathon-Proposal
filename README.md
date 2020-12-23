# Blind U-Net for Secure COVID-19 CT Segmentation

## Motivation

Recently, deep neural networks (DNNs) have been widely used in medical applications (e.g., medical image computing and disease diagnosis) because of their high eﬃciency and extremely prediction accuracy. However, the strict security regulations placed on medical records hinder the use of big data in machine learning (ML). Access controls and client-side encryption are mandated for the distribution of patient records over public network. In order to deploy DNN models to the real-world for medical image analysis, the key problem is how to handle the data transfer and computation securely and eﬃciently. This leads to our key motivation, we propose a secure protocol for the UNET architecture, named blind UNET (BUNET), that enables input-hiding segmentation on medical images. In the proposed protocol, we use a combination of cryptographic building blocks to ensure that client-side encryption is enforced on all data related to the patients, and that practical inference time can also be achieved. As a result, medical institutions can take advantage of third-party machine learning service providers without violating privacy regulations. 

## Use Scenario

We use the application of ML-based CT-based COVID-19 diagnosis. In this scenario, we suppose Alice (a doctor in a hospital) is the client who will provide the patients’ health records such as CT scan, age, gender to Bob (a third-party machine-learning service providers) for medical analysis. These patient data will be used to analyze and infer the detection of COVID-19 and quantify the infectious region. The client can get to know whether this patient has COVID-19 as well as the infectious progressive stage.

## Security Protocol

 - lattice-based packed additive homomorphic encryption (PAHE)
 - additive secret sharing (ASS) 
 - garbled circuits (GC)
 - multiplication triples (MT) schemes

## How to use our code

1. Download the data from this [link](https://drive.google.com/drive/folders/1g2YDMp4wCALeQLBXi-NsY6eW_wExfHO-?usp=sharing) and the pretrained model from this [link](https://drive.google.com/drive/folders/1ASbqSiKx7d1m1nvSW6h0dba_HlIXA_oq?usp=sharing).
2. Run the predition command and you will get segmentation result in ./result folder
```
python prediction.py --model_infection_dir YOUR_MODEL_INFECTION_DIR --model_lung_dir YOUR_MODEL_LUNG_DIR --test_data_dir YOUR_TEST_DATA_DIR
```

### Input and Output

- **Input**, we use the 3D CT volume (512x512x512) as the input of our system.
- **Output**, the output of the system is a 3D binary segmentation mask of the corresponding CT image. Regions with pixel value 1 mean these areas are infected with COVID-19, Regions with pixel value 0 are normal tissues.
  
An example of the segmentation result can be seen in the figure below. The left figure shows the original slice (extracted from a 3D CT volume), the right figure shows the segmentation result. Red means the region is infected.

![example](/images/example.png)

### Performance Evaluation

Our work is based on the pretrained model from this paper: **A Rapid, Accurate and Machine-Agnostic Segmentation and Quantification Method for CT-Based COVID-19 Diagnosis** [link](https://ieeexplore.ieee.org/abstract/document/9115057). We modified the parameters of the original model and add quantization to them in order to reduce the computation overhead. We also add security protocol on top of the network architecture. The security protocols we use (e.g., PAHE and GC) are standard protocols, which are not included in our code. You can refer to [PAHE](https://github.com/snucrypto/HEAAN) and [GC](https://github.com/ojroques/garbled-circuit) if you are interested in their detailed implementation. 

#### What to evaluate?

##### Accuracy
Since the ground truth label of the CT volume is not provided by the original paper, we use their pretrained model to generate segmentation masks on the testset and use these masks as the ground truth baseline of our method. We need to evaluate **how much the accuracy drops** once we apply the security protocol in the inference phase. Therefore, we report the segmentation accuracy as Dice coefficient in our prediction.py script. The Dice coefficient of our baseline (no quantization and security protocol) is 1.0.

##### Runtime
Existing secure inference solutions are usually very slow, we hope to design a protocol that can reduce the inference computation overhead and latency. Therefore, the runtime is also an important evaluation metric for our work. Since we break down the 3D segmentation into three 2D segmentation along each axis, we tested and reported the runtime of inference a single 2D image. The total runtime is the sum of all 2D inference.

