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

- X_inbred : one-hot matrix of inbred ID with the dimension of `m-by-nb`, where `m` is the number of observations and `nb` is the number of inbreds.

- X_tester : one-hot matrix of tester ID with the dimension of `m-by-nt`, where `m` is the number of observations and `nt` is the number of testers.

- X_inbred_cluster : one-hot matrix of inbred genetic ID with the dimension of `m-by-ncb`, where `m` is the number of observations and `ncb` is the number of genetic groups for inbreds.


- X_tester_cluster : one-hot matrix of inbred genetic ID with the dimension of `m-by-nct`, where `m` is the number of observations and `nct` is the number of genetic groups for testers.


- X_loc : one-hot matrix of location ID with the dimension of `m-by-nl`, where `m` is the number of observations and `nl` is the number of planting locations.

- Yield : the vector of response variables (yield) with the dimension of `m-by-1`.

For more information, please see the data section of the paper.


## Data Availability

The data analyzed in this study was provided by Syngenta for 2020 Syngenta Crop Challenge. We accessed the data
through annual Syngenta Crop Challenge. During the challenge, September 2019 to January 2020, the data was open to
the public. Researchers who wish to access the data can send their request to Syngenta through <a href="https://www.ideaconnection.com/syngenta-crop-challenge/contact.php" target="_blank">this website</a>.


 


