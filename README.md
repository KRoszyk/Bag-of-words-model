# Bag of words model for famous places in Poznań
In this project I implemented machine learning model to recognize five different buildings in Poznań. 

They are: UAM, Katedra, Okrąglak, Baltyk and Teatr Wielki.

# Bag of words image classifier - what is that?

Bag of words is a famous method for Natural Lanugage Processing. 
This method is based on the creation of a so-called vocabulary, which contains characteristic features of the processed data. 
Then, these features are grouped into histograms to determine how similar the data from the training and test sets are.

Hovewer, idea of Bag of words isn't only used for NLP. It is also very useful for image classification, 
where the counterparts of the words are the features detected by the feature detector-descriptor - for example SIFT. 
The crteated descriptors are then clustered to form histograms for each image. The last step is to classify the histograms and finally assign the images to the labels that best suits them.

An illustration of how the bag of words model works is presented below.

![Screenshot](https://github.com/KRoszyk/Bag_of_words/blob/master/images/illustration.jpg)

If you need more explanations, you can find them at the link below:
https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj4/html/vdedhia6/index.html

# What's in this repository?

This repository contains 2 directories divided into 5 categories. **Train** folder includes train images and **test** folder includes test images.

Additionally, a file **change.py** was created to remove images with Polish characters in their names, because the **OpenCV** library has some problems with reading them.

Moreover, the training set was enlarged using **data augmentation techniques**. This allowed me to increase the amount of training data and make the model more resistant to extreme classification cases.
The image augmentation technique was performed with the **Augmentator** library, and the method responsible for augmentation is presented below.

```python
def make_augmentations():
    p = Augmentor.Pipeline("./../")  # put here the path to your train images
    p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.3, percentage_area=0.8)
    p.flip_top_bottom(probability=0.1)
    p.sample(500)
   ```
   
The main file responsible for creating the vocabulary and classifier models is **main_code.py**. This file includes some methods like for example traning model using kmeans clusterization, training classifier and creating histograms for images. 
   
# Selected detector and classifier

In order to create an efficient image classification system, I decided to choose the **SIFT** detector and the **SVC** classifier. They were selected experimentally due to the time of data processing and accuracy.

What is more, in order to optimize the classifier, the **Grid Search** algorithm was used to ensure the best of the given and available parameters during the classification.

The vocabulary size of 128 words was also selected experimentally. It should be remembered that too large vocabulary may affect the model overfitting and the appearance of a bias.

Finally, in order to facilitate model testing, the dictionary and classifier were packed with the **pickle** tool. 

# Summary

Below is the classification result for the test set. As you can see, this is an accuracy of 90%, which is a quite good result for the classic machine learning method.
```
Result on testing images: 0.9
Time of testing the model: 00:00:11
```

If you want to test a learned model, just call the **test_model()** method. 
If you would like to add another building class or change the types of classified objects, remember to retrain the models with **train_model()** method!
