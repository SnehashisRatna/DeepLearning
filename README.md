# DeepLearning
## Submitted By  
- **Name:** Snehashisa Subudhiratna  
- **Roll Number:** A124013
- **Course:** M.Tech in Computer Science Engineering  
- **Institute:** IIIT Bhubaneswar  
- **Subject:** Deep Learning    

# Assignment 3: Perceptron Implementation for Logic Gates  

## Overview  

This assignment implements a Perceptron model to classify the outputs of basic logic gates (AND, OR, NOT). The implementation includes:  
- A custom Perceptron model built from scratch using NumPy.  
- Scikit-learn’s built-in Perceptron for comparison.  
- Visualization of decision boundaries for AND, OR, and NOT gates using Matplotlib.  

## Repository Structure  

- `LogicGateOperation.py` &nbsp;&nbsp;&nbsp;&nbsp;: Main Python script containing Perceptron implementations and plotting functions.  
- `README.md` &nbsp;&nbsp;&nbsp;&nbsp;: This file.  

## Requirements  

To run the code, ensure you have Python 3 installed along with the following libraries:  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Scikit-learn](https://scikit-learn.org/)  
## Code Link 

You can view the complete code on GitHub:
[Assignment 3 Code](https://github.com/SnehashisRatna/DeepLearning/blob/main/LogicGateOperation.ipynb)

## Submission Date
*7th-Feb-2025*

# Assignment 4: Perceptron Classifier on the Iris Dataset
 ## Overview
  This assignment implements a Perceptron classifier using both:
  - A custom Perceptron implementation (from scratch)
  - Scikit-learn’s built-in Perceptron

 The goal is to perform binary classification on a subset of the Iris dataset (setosa vs. non-setosa) and visualize the decision boundaries.
 
 ## Repository Structure
 
 - `irisDataSet.py` &nbsp;&nbsp;&nbsp;&nbsp;: Main Python script containing the Perceptron implementations and plotting functions.
 - `README.md` &nbsp;&nbsp;&nbsp;&nbsp;: This file.
 
 ## Requirements
 
 To run the code, you need to have Python 3 installed along with the following libraries:
 - [NumPy](https://numpy.org/)
 - [Matplotlib](https://matplotlib.org/)
 - [Scikit-learn](https://scikit-learn.org/)
 
 
 
 Upon execution, two decision boundary plots will be generated:
 - One for the custom Perceptron implementation.
 - One for the scikit-learn Perceptron implementation.
 
 ## Code Link
 
 You can view the complete code on GitHub:
 [Assignment 4 Code](https://github.com/SnehashisRatna/DeepLearning/blob/main/IrisDataSet.ipynb)

   ## Submission Date
   
   *7th-Feb-2025*

# NoteBook - Chapter-03

# **Shallow Neural Networks I**

## **Overview**  
This notebook is designed to help you gain familiarity with **shallow neural networks** using **1D inputs**. It follows an example similar to **Figure 3.3** and allows experimentation with different **activation functions**.  

The key objectives of this notebook are:  
- Understanding the structure of a shallow neural network.  
- Implementing different **activation functions**.  
- Modifying and tuning parameters to observe their effects.  
- Computing **loss functions**, including least squares error and negative log likelihood.  

## **Tasks in the Notebook**  
- **Implementing Activation Functions:** Experiment with different activation functions like ReLU, Sigmoid, and Tanh.  
- **Defining Network Parameters:** Initialize weights and biases for a shallow neural network.  
- **Forward Propagation:** Compute the output using matrix multiplications and activation functions.  
- **Probability Distribution:** Implement the **Gaussian distribution** function and analyze its behavior.  
- **Loss Computation:** Compute and visualize **sum of squares loss, likelihood, and negative log-likelihood**.  
- **Multiclass Classification:** Implement softmax and analyze the model’s behavior in classification tasks.  

## **Expected Outputs & Observations**  
- The model should fit 1D data points and make reasonable predictions.  
- Changing activation functions affects the shape of the output.  
- Modifying parameters impacts **likelihood and loss values**.  
- The softmax function should produce **probability distributions** over class labels.

# **Shallow Neural Networks II**  

## **Overview**  
This notebook is designed to help you gain familiarity with **shallow neural networks** using **2D inputs**. It follows an example similar to **Figure 3.8** and allows experimentation with different **activation functions**.  

The key objectives of this notebook are:  
- Understanding the structure of a shallow neural network with **2D inputs**.  
- Implementing different **activation functions**.  
- Modifying and tuning parameters to observe their effects.  
- Computing **loss functions**, including least squares error and negative log likelihood.  

## **Tasks in the Notebook**  
- **Implementing Activation Functions:** Experiment with different activation functions like ReLU, Sigmoid, and Tanh.  
- **Defining Network Parameters:** Initialize weights and biases for a shallow neural network.  
- **Forward Propagation:** Compute the output using matrix multiplications and activation functions.  
- **Probability Distribution:** Implement the **Gaussian distribution** function and analyze its behavior.  
- **Loss Computation:** Compute and visualize **sum of squares loss, likelihood, and negative log-likelihood**.  
- **Multiclass Classification:** Implement softmax and analyze the model’s behavior in classification tasks.  

## **Expected Outputs & Observations**  
- The model should fit **2D data points** and make reasonable predictions.  
- Changing activation functions affects the shape of the output.  
- Modifying parameters impacts **likelihood and loss values**.  
- The softmax function should produce **probability distributions** over class labels.  

# **Shallow Neural Networks III & IV**  

## **Overview**  
This repository contains notebooks designed to help you gain familiarity with **shallow neural networks**, covering both **2D inputs** and the **maximum possible number of linear regions**. It also explores different **activation functions** and their effects on network performance.  

The key objectives of these notebooks are:  
- Understanding the structure of a shallow neural network with **2D inputs**.  
- Implementing different **activation functions**.  
- Modifying and tuning parameters to observe their effects.  
- Computing **loss functions**, including least squares error and negative log likelihood.  
- Analyzing the **maximum possible number of linear regions** a shallow network can form.  

## **Notebook 3.3: Shallow Network Regions**  
- **Computing Linear Regions:** Compute the **maximum possible number of linear regions** in a shallow network, as seen in Figure 3.9 of the book.  
- **Parameter Effects:** Modify parameters and analyze their impact on network regions.  
- **Mathematical Analysis:** Understand the mathematical reasoning behind network region calculations.  

## **Notebook 3.4: Activation Functions**  
- **Exploring Activation Functions:** Implement and compare different activation functions.  
- **Effect on Model Output:** Observe how each activation function affects model predictions.  
- **Gradient Behavior:** Analyze gradients of activation functions and their impact on training.  

## **Expected Outputs & Observations**  
- The model should fit **2D data points** and make reasonable predictions.  
- Changing activation functions affects the shape of the output.  
- Modifying parameters impacts **likelihood and loss values**.  
- The softmax function should produce **probability distributions** over class labels.  
- The number of linear regions should follow a predictable pattern based on theoretical calculations.  




