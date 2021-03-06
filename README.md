# Scene Recognition

Classifying images in three ways, using machine learning:
- run1: k-nearest-neighbours, OpenImaj
- run2: linear classifiers, OpenImaj
- run3: InceptionV3 pretrained model on Imagenet, Python

## Run1

To implement the tiny image K-nearest-neighbours classifier first we iterated through every image in every group, extracting the feature vector. This involved making the image square by taking the size of its smallest dimension and then cropping the image to a square of this size. This image was then resized to the suggested resolution of 16x16. Finally, this image was normalised using a built-in OpenIMAJfunction, producing the feature vector.
These feature vectors are then used to construct/train an instance of the DoubleNearestNeighboursExact class. Using this object, we can then provide it an image to classify, specifying the K number of neighbours to consider. This returns a list consisting of the index of the K-nearest feature vectors and their corresponding distances. Using the indexes, we matched each feature vector to their corresponding group. With the groups known, we could then count the number of “votes” for each group, returning the group with the majority as the classification.
To test the classifier, we split the training dataset in an 80:20 ratio, reserving 20% for testing. To determine the optimal K-value we calculated the average accuracy across a range of possible values. This showed that a K-value of 5 produced the best results, with an average accuracy of 22.3%.

## Run2

After loading the dataset, we have split the provided dataset into training and testing, with an 80 to 20 ratio. The next stage was getting a sample of random patches from each image. As recommended in the coursework description, we kept the size of the patches to 8x8 and the sampling rate to 4 in both x and y directions. From the patches collected, first, a rectangular region of interest was gathered using the .extractROI method. The pixels from the resultant FImage were transformed into a 1D vector and then into a feature vector. The spatial location of the feature was assigned as the x and y of the original patch. Using the values of spatial location and the feature vector, a local feature is generated. This is then done for all of the patch samples. To learn the vocabulary, we have used K-Means clustering with 500 clusters. We did not observe a significant change in the accuracy when the cluster value or the number of sample patches was changed.

The Extractor class we defined uses BagOfVisualWords which assigns each feature with a visual word using the HardAssigner class. Then the histograms are computed using BlockSpatialAggregator. Lastly, the histograms were combined together and normalised. By using the parameters provided in Chapter 12, we created and trained the linear classifier.  The accuracy was obtained using the OpenIMAJ evaluation framework.

Average accuracy: 26.7%

## Run3

For this part, we have decided to use a deep learning framework to classify the testing images in 15 classes. We used Python, Keras and Tensorflow to implement our solution. For us to have a better understanding of the performance of the algorithms we developed, we used 20% of the training images for testing.

Throughout the development of this part, different strategies have been tried. The first one was to train a neural network from scratch containing three convolutional layers, two max pooling layers, and three dropout layers. We also used data augmentation by flipping the images horizontally and randomly rotating them with a factor of 0.1. The network performed better on the training data but poorly on the testing data, meaning that it was overfitting. Even with the regularizations of dropout layers and augmenting the data, the performance of this network was very low, roughly 56%.

We came to the conclusion that we cannot train a neural network from scratch with the small dataset that we have (100 samples per class) in order to get higher accuracy. Therefore, the second strategy, and final solution, was to use transfer learning meaning that we worked on top of a pre-trained neural network. We tried different experiments with VGG16 and InceptionV3 (state-of-the-art neural networks). We used the feature extraction layers from the pre-trained neural networks with the weights from Imagenet to extract the features from our images and we constructed the model for the last fully connected dense layers which can be seen in Figure 1. We trained the final model by using a batch size of 64 and 50 epochs. This decision was made after fine-tuning the hyperparameters and deciding these are the best fits.

On the VGG16 we tested with an image size of (64,64) and got an accuracy of approximately 70%, while with the InceptionV3 the performance of the algorithm with a smaller image size was lower (approx. 60%). On InceptionV3 we tested with the initial image size (255,255) and the accuracy increased dramatically, reaching 90% accuracy. A downside to this improvement was that the computational power increased with the size of the target image.

The final solution uses transfer learning with InceptionV3, which requires an image of size (255,255) and uses a batch size of 64 and 50 epochs for fitting the model. In the end, the model was trained on the entire data set and produced the class predictions of each image from the testing folder, but the model trained on the training data split into testing and training, the accuracy was approximately 90%. We expect that the accuracy will increase now that we used the entire training data to train the model.

![fig1](https://user-images.githubusercontent.com/37437735/147774039-c5dd7fb7-19af-419a-9d0e-c3a4674008c0.PNG)
