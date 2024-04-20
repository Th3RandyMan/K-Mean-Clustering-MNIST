# K-Mean Clustering Dictionary
EEC 289A Assignment 1

## Overview
For this project, patches will be extracted from an image to represent relatively small sections that make up different geometric shapes. With these patches, K-Means will be applied to cluster similar patches into groupings. The center of each grouping, the centroids, will compose the image dictionary as each centroid can be used as a building block for the different shapes of the images.

MNIST is the dataset used in this project. By training an image dictionary, the images can be represented as encoded vectors that reference patches in the dictionary. Images can then be compressed, reconstructed, and denoised using this technique. 

## Requirements
* Numpy
* Jupyter Notebook
* Matplotlib
* Scikit

## Results
With a relatively small number of centroids, shapes will begin to blur as the tolerance for similar patches in a cluster will change. Less centroids lead to larger clusters that may relate patches that are not very similar. To read more about the results, there is a paper saved in this repository discussing the topic further.

## Acknowledgements
This project is a course assignment for an unsupervised learning class taught by Yubei Chen, Spring 2024. 
