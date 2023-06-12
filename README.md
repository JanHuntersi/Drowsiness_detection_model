# Drowsiness drivers-fatique Detection AI Implementation

This here is a Jupyter notebook implementation written in Python programming language for fatiquesness or drowsiness detection with a machine learning model.

This file contains code for image preprocessing and loading large datasets. Initializing and training the model and optimizing it with training it with different parameters and increasing the dataset via image augmentation.

## What are we detecting
Our model is going to be able to classify images in to 4 categories. Closed -> for closed eyes, no_yawn-> person is not
yawnimg, open-> person has open eyes and finally  yawn-> person is yawning.

## Model
In the  init_model() function we build our machine learning model. It is a sequential model with a 2D convolutional layer, max pooling layer, flatten layer,dropout layer and dense layer. The output layer has 6 units and softmax activation for multi-class classification- used for our 4 categories.

## Libraries Used
- os: Used for directory and file operations
- cv2: Used for image processing
- numpy: Used for numerical computations
- matplotlib.pyplot: Used for image visualization
- random: Used for random operations
- tensorflow: Used for machine learning purposes
- sklearn.metrics: Used for confusion matrix calculation
- sklearn.model_selection: Used for train-test split of dataset
- scipy.ndimage: Used for image rotation
- keras: Used for model building

## Dataset 
- used the yawn_eye_dataset_new, which is available [Kaggle](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new)

![image](https://github.com/JanHuntersi/SEMINARSKA_NALOGA/assets/55513538/ad8184e1-38c6-44b4-98ac-30405faa2028)
![image](https://github.com/JanHuntersi/SEMINARSKA_NALOGA/assets/55513538/ea8762d5-8a42-408e-93bd-324f8cead358)
![image](https://github.com/JanHuntersi/SEMINARSKA_NALOGA/assets/55513538/2c6d3265-0f60-4395-a3dd-b26a304de81e)

