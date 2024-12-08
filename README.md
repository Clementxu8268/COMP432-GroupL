# COMP432_Project_Group_L
## Overview
This project addresses the challenge of colorectal cancer classification using deep learning and feature analysis. It is divided into two main tasks that focus on training, analyzing, and applying convolutional neural networks (CNNs) for feature extraction and classification. This project combines deep learning with advanced visualization and classical machine learning to provide an in-depth analysis of colorectal cancer data across multiple datasets. It offers insights into CNN-based feature extraction and enables meaningful comparisons between custom-trained and pre-trained models.The pipeline incorporates transfer learning, t-SNE visualization, and classical machine learning classification techniques. 



## 1. Dataset Download and Check Setup

### **Dataset Setup Instructions**
The project requires three datasets to be manually downloaded and placed in the root directory.

#### **Steps to Download and Setup:**
1. Download the dataset FOLDERS directly from the following URLs:
   - [Dataset 1](https://drive.google.com/drive/folders/1t3KajrWrV756rKBe0H2-qXxUG-e3tQBF?usp=drive_link)
   - [Dataset 2](https://drive.google.com/drive/folders/15tEA6I0YETiZjBa0ACdWzWXGG0lBl2fm?usp=drive_link)
   - [Dataset 3](https://drive.google.com/drive/folders/12OqwIkygpRoYVtNo-oBfvfnTuNSSy32b?usp=drive_link)

2. Extract the datasets into the ROOT directory of this project. After extraction, the directory structure should look like this:

   my_project/

   ├── Dataset1/

   ├── Dataset2/

   ├── Dataset3/

   └── other_project_files/

## 2. Tasks
   **Task1: Colorectal Cancer Classification**
   - Train a CNN model (ResNet 18) on Dataset 1 for colorectal cancer classification.
   - Use t-SNE to visualize the extracted features of the encoder and analyze the feature separability across classes.
   - Report training results in terms of accuracy and loss.

   **Task2: Feature Analysis and Transfer Learning**
   - Apply the trained CNN encoder from Task 1 (without its classification head) to Dataset 2 and Dataset 3.
   - Analyze and visualize extracted features using t-SNE.
   - Repeat the above using a pre-trained ImageNet encoder.
   - Perform a classical machine learning classification (Random Forest) on the extracted features for Dataset 2 and Dataset 3.

## 3. Prerequisites 

The following Python libraries are required to run this project:

pip install torch torchvision numpy matplotlib os pathlib sklearn

## 4. Instructions on how to train/validate the model
**Task1: Dataset1**  
The dataset1 is located in the directory ./Dataset1/ColorectalCancer and contains three classes:
- MUS 
- NORM
- STR 

   **Run the Notebook for task1: task1.ipynb**  
   Execute the cells step-by-step. Key sections include:
   
   1. Data Preprocessing and Splitting: Loads and preprocesses images, splits into training and validation sets.
   2. Model Definition: Defines a custom ResNet-18 model architecture.
   3. Training: Includes training and validation loops, with real-time plots of accuracy and loss.
   4. Feature Extraction: Uses the trained model to extract features and visualize them with t-SNE.
   5. Evaluation: Generates confusion matrices and classification reports.

If training is skipped,  you can directly use the pre-saved model in Task 2, named **final_feature_resnet18.pth**.

**Task2: Dataset2 and Dataset3**  
The dataset2 is located in the directory ./Dataset2/ProstateCancer and contains three classes:
- gland
- nongland
- tumor
  
   **Run the Notebook for task2: task2_dataset2.ipynb**     
   Execute the cells step-by-step. Key sections include:  
   1. Load a saved ResNet-18 model and applies it to Dataset 2 (Prostate Cancer) for feature extraction. The extracted features are visualized in 2D using t-SNE to analyze class separability.
   2. Generates confusion matrices.
   3. Generates classification reports.
 
The dataset3 is located in the directory ./Dataset3/AnimalFaces and contains three classes:
- cat
- dog
- wild

     **Run the Notebook for task2: task2_dataset3.ipynb**   
   Execute the cells step-by-step. Key sections include:  
   1. Load a saved ResNet-18 model and applies it to Dataset 3 (Animal Faces) for feature extraction. The extracted features are visualized in 2D using t-SNE to analyze class separability.
   2. Generates confusion matrices.
   3. Generates classification reports.


## 5. Instructions on how to run the pre-trained model  
This script uses a pre-trained ResNet-18 model to extract features from two datasets: Dataset 2 (Prostate Cancer) and Dataset 3 (Animal Faces). The extracted features are visualized in 2D using t-SNE, showcasing the separability of different classes within each dataset.  

- The dataset2 is located in the directory ./Dataset2/ProstateCancer and contains three classes:gland, nongland, tumor.
- The dataset3 is located in the directory ./Dataset3/AnimalFaces and contains three classes:cat, dog, wild.
  
   **Run the Notebook**       
   Execute the cells step-by-step.   
   Key sections include:  
   1. Load a pre-trained ResNet-18 model and applies it toDataset 2 (Prostate Cancer) and Dataset 3 (Animal Faces) for feature extraction. The extracted features are visualized in 2D using t-SNE to analyze class separability.
   2. Generates confusion matrices.
   3. Generates classification reports.
