# Task 1: Semi-supervised Learning - Self/Co-Training

The task was to use Wikipedia API to extract the summaries of random wikipedia articles and label a part of them as either STEM or NON-STEM subjects. About 30 - 40 labels. The remaining would be unlabelled.

A semi-supervised learning model was to be made. It basically works by training a model on the labelled set and using that model to predict on the remaining dataset. The model also has a confidence threshold for example 70%. Such that it will only add the labels to the remaining set if it is atleast 70% sure that it's prediciton is correct. This process is reapeated a certain number of times until we have a fully trained model.

I used Sklearn library which has a method to train selflearning models. It takes a base estimator model and uses that to turn it into a selflearning model.

Note that it requires the unlabelled part of the dataset to be set to -1.
