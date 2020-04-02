# Neural collaborative filtering for crop yield prediction

This repository contains my codes for the paper titled "Predicting Yield Performance of Parents in Plant Breeding: A Neural Collaborative Filtering Approach".


<a href="https://arxiv.org/abs/2001.09902" target="_blank">"Predicting Yield Performance of Parents in Plant Breeding: A Neural Collaborative Filtering Approach"</a>. The paper was authored by Saeed Khaki, Zahra Khalilzadeh, and Lizhi Wang.

In the 2020 Syngenta Crop Challenge, Syngenta released several large datasets that recorded the historical yield performances of
around 4% of total cross combinations of 593 inbreds with 496 testers which were planted in 280
locations between 2016 and 2018 and asked participants to predict the yield performance of cross
combinations of inbreds and testers that have not been planted based on the historical yield data
collected from crossing other inbreds and testers. In this paper, we present a collaborative filtering
method which is an ensemble of matrix factorization method and neural networks to solve this
problem.



## Getting Started 

Please install the following packages in Python3:

- numpy
- tensorflow
- matplotlib


## Dimension of Input Data

- X_inbred: The one-hot matrix of inbred ID with the dimension of `m-by-nb`, where `m` is the number of observations and `nb` is the number of inbreds.

- X_tester: The one-hot matrix of tester ID with the dimension of `m-by-nt`, where `m` is the number of observations and `nt` is the number of inbreds.



 


