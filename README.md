# crowd_crush_detection
A repository to detect and monitor points of potential crowd crush

- Link to Demo Video: https://www.youtube.com/watch?v=6tSz3sny-KI&ab_channel=SiddarthVinnakota
- Link to Presentation Slides: https://docs.google.com/presentation/d/1hHDwBKbd7i4YkKZXi6DmN4CpLCe4BC1ywxt9JSCRRyg/edit?usp=sharing

## Problem and Motivation
Crowding is common amongst large gatherings such as concerts, parties, or other religious events. An unchecked crowd can compress on certain points, leading to panic, injuries, and even death. Current keypoint detection approaches like YOLO are ineffective when dealing with large swaths of people, where the individual points are blended together. Instead, what if we could train the model on crowd and sparsity patterns?

## Data Sourcing and Data Processing
This dataset is custom created via Unity. A stage is constructed, and 30 individuals will be randomly shuffled, getting their individual ground truth positions. Afterwards, a script (make_heatmaps.py) is ran creating density maps used via Gaussian blobs.
The scripts are designed to be replicable and adjustable when it comes to the location and the number of individuals. The images are further split into train, test, and validation folders. The main reason why I created my own dataset is not only for ethical considerations, but I also have complete access to the ground truth positions. 

The Kaggle Dataset can be found here: https://www.kaggle.com/datasets/siddarthvinnakota/generated-crowds/data 

## Relevant Previous Efforts
- YOLO-CROWD (https://github.com/zaki1003/YOLO-CROWD): A lightweight crowd counting and face detection model that is based on Yolov5
- CrowdNet (https://arxiv.org/abs/1608.06197): Uses a combination of deep and shallow networks and then concatenates them.
  - NOTE: The deep learning model I constructed also uses the name "CrowdNet" by accident. This model has no direct affiliation otherwise.
- CSRNet (https://arxiv.org/abs/1802.10062): Uses multiple VGG-16 Convolutional Layers as the backbone. Can be paired with a decoder for upsampling

## Model Evaluation Strategy and Metric
Python script (makeheatmaps.py) that uses the ground truth json file to create a density map via Gaussian blobs. Our approaches will attempt to predict a density map via regression-based models. It will be compared via the Mean Squared Error (MSE)

## Naive Approach
- Regardless of the image, randomly places 30 points on a matrix the same size as the image and uses a Gaussian blob to predict a matrix
- MSE of 0.1168
- Good for this dataset, but it is not scalable.

## Machine Learning Approach (SGDRegressor with Local Features)
- For every image:
  - Pass through a sliding window of 16x16 pixels through both the image and corresponding heatmap
  - For each window in the image extract the Gabor Features
  - The Corresponding window of pixels in the heatmap gets the max pixel value as our ground truth labels
- The extracted data will be scaled, applied an RBF kernel transformer, and ran through an SGDRegressor to predict the max window density at every 16x16 sliding window
- Unfortunately Performs Extremely poorly (an MSE of ~1500)

## Deep Learning Approach ("CrowdNet")
- A Two Part System
  - Encoder: Uses ResNet18 as the backbone due to it being lightweight and uses already learned features.
    - Could add more layers, as long as we don’t include the last 2 layers (classification based)
  - Decoder: A series of convolutional layers and upsampling to compress and expand the image until we get our density heatmap
- MSE of 0.0059

  ![CrowdNet](https://github.com/user-attachments/assets/35ae112e-b566-405b-ad70-99f88017ba0e)

## Model Comparisons

| Approach              | MSE   |
|-----------------------|-------|
| Naive Approach        | 0.1168  |
| Machine Learning| ~1500   |
| Deep Learning         | 0.0059   |

## Results and Conclusions
- The Naive Model is good, yet is not scalable to more points.
- The Machine Learning Approach needs more fine-tuning and model-building to achieve a reasonable MSE.
- The Deep Learning model is the best performing model, and is scalable.

### Future Work
- Configure more datasets with more varied crowds and environments
- Improve upon the Machine Learning Approach
- Test the Deep Learning model on public crowd datasets to increase robustness

## Ethics Statement
- "CrowdNet” and the other approaches were trained using synthetically generated data which is designed to imitate crowd scenarios. It is important to note that it is still in active development and not ready for production
- Predictions should not be considered as the end-all-be-all without additional human verification, especially in more high stakes situations
- I am committed to responsible AI development and I showcase the transparency and limitations of these approaches. Synthetic data helps ensure data privacy, but real world training is still necessary to ensure robust deployment.

## Github Structure

```
crowd_crush_detection/
│
├── Dataset/ # Folder containing sample image, json, and heatmap sets. It is also premade into train, testing, and validation subfolders as well
|   ├── crowd_heatmaps
|   ├── crowd_images
|   ├── crowd_images
│   ├── train/
│   ├── val/
│   └── test/
|
├── data_collection_scripts/ # A series of C# scripts used in Unity to generate the dataset
|   ├── CameraCapture.cs
|   ├── RandomPos.cs
│   └── ShuffleManager.cs
│
│
├── models/                       # Pretrained models or other pickle files
|   ├── rbf_feature_transformer.pkl
|   ├── scaler.pkl
|   ├── sgd_rbf_model.pkl
│   └── best_crowdnet.pth
├── sample_images/ #Some sample images that I found worked best with my frontend
│
├── dl_approach.py                     # Deep Learning Neural Network Approach
├── make_heatmaps.py               # Custom Python Script to convert the ground truth json files into density map images
├── ml_approach.py                  # Classical Machine Learning Approach
├── naive_approach.py               # A simple Naive Baseline
├── streamlit_app.py                   # Function to run the streamlit frontend
├── requirements.txt                # Python dependencies
├── README.md                       # Project description and usage guide
└── LICENSE                         # License information
```

## Instructions on running it

The sample data is there and already formatted, so no need to run make_heatmaps.py. Instead, you can call the different approaches directly to train and test a model, using the sample Dataset


