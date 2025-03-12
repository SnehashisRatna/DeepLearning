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

## Code Link 

You can view the complete code on GitHub:
[Shallow Neural Networks I Code]([https://github.com/SnehashisRatna/DeepLearning/blob/main/LogicGateOperation.ipynb](https://github.com/SnehashisRatna/DeepLearning/blob/main/NoteBook(Chap-03)/3_1_Shallow_Networks_I.ipynb))

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
## Code Link 

You can view the complete code on GitHub:
[SNN 2 Code](#[https://github.com/SnehashisRatna/DeepLearning/blob/main/LogicGateOperation.ipynb](https://github.com/SnehashisRatna/DeepLearning/blob/main/NoteBook(Chap-03)/3_2_Shallow_Networks_II.ipynb))

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

## Code Link 

You can view the complete code on GitHub:
[SNN 3 Code]([https://github.com/SnehashisRatna/DeepLearning/blob/main/LogicGateOperation.ipynb](https://github.com/SnehashisRatna/DeepLearning/blob/main/NoteBook(Chap-03)/3_3_Shallow_Network_Regions.ipynb))

## **Notebook 3.4: Activation Functions**  
- **Exploring Activation Functions:** Implement and compare different activation functions.  
- **Effect on Model Output:** Observe how each activation function affects model predictions.  
- **Gradient Behavior:** Analyze gradients of activation functions and their impact on training.  

## Code Link 

You can view the complete code on GitHub:
[SNN 4 Code]([https://github.com/SnehashisRatna/DeepLearning/blob/main/LogicGateOperation.ipynb](https://github.com/SnehashisRatna/DeepLearning/blob/main/NoteBook(Chap-03)/3_4_Activation_Functions.ipynb))

## **Expected Outputs & Observations**  
- The model should fit **2D data points** and make reasonable predictions.  
- Changing activation functions affects the shape of the output.  
- Modifying parameters impacts **likelihood and loss values**.  
- The softmax function should produce **probability distributions** over class labels.  
- The number of linear regions should follow a predictable pattern based on theoretical calculations.  


# **Deep Neural Networks & Advanced Concepts**  

## **Overview**  
This repository contains notebooks that explore deeper neural network architectures, covering topics such as **composing networks**, **clipping functions**, and **deep neural networks in matrix form**. These notebooks extend shallow networks by stacking multiple layers and analyzing their transformations.

### **Objectives:**  
- Understanding how neural networks behave when composed together.  
- Experimenting with **clipping functions** and how they modify representations.  
- Converting **deep neural networks** to matrix form for efficient computation.  
- Observing the effects of different activation functions and layers.  

---  

## **Notebook 4.1: Composing Networks**  
- **Composing Multiple Networks:** Feeding one network’s output into another.  
- **Network Variability:** Experiment with different network configurations.  
- **Observing Effects:** Predict outcomes before running experiments.  

## **Notebook 4.2: Clipping Functions**  
- **Understanding Hidden Layers:** Analyze how multiple hidden layers clip and recombine representations.  
- **Building Complex Functions:** Experiment with stacking different activation functions.  
- **Observing Function Approximations:** Modify network parameters to shape output behavior.  

## **Notebook 4.3: Deep Neural Networks**  
- **Matrix Form Representation:** Convert deep neural networks into efficient matrix operations.  
- **Scaling Networks:** Understand computational efficiency in deep architectures.  
- **Training & Optimization:** Analyze gradient flow and parameter updates.  

## **Expected Outputs & Observations**  
- Composed networks should exhibit changes in function approximations.  
- Clipping functions modify how intermediate representations are formed.  
- Deep networks should efficiently process inputs when converted to matrix form.  
- Different activation functions impact learning speed and convergence.  

## **Code & Resources**  
📂 **Repository Link:** [Chapter-4](https://github.com/SnehashisRatna/DeepLearning/tree/main/Notebooks/Chap04)
📜 **Reference Material:** UDL Book - Chapter 4  
✉️ **Contact:** udlbookmail@gmail.com  


# **Deep Neural Networks & Loss Functions**

## **Overview**  
This repository contains notebooks designed to explore **deep neural networks** and different **loss functions**, including **Least Squares Loss**, **Binary Cross-Entropy Loss**, and **Multiclass Cross-Entropy Loss**. These practicals follow the theoretical concepts from **Chapter 5** of the book.

### **Key Objectives:**  
- Understanding the **Least Squares Loss** and its connection to **Maximum Likelihood Estimation**.  
- Implementing **Binary Cross-Entropy Loss** for classification problems with binary labels.  
- Implementing **Multiclass Cross-Entropy Loss** for multi-class classification.  
- Computing **log-likelihood functions** and their gradients for optimization.  


## **Notebook 5.1: Least Squares Loss**  
- **Mathematical Derivation:** Understand the equivalence of **Maximum Likelihood** and **Negative Log-Likelihood Minimization**.  
- **Implementation:** Write Python functions to compute the **Least Squares Loss**.  
- **Gradient Descent:** Compute gradients and visualize the loss surface.  

## **Notebook 5.2: Binary Cross-Entropy Loss**  
- **Bernoulli Distribution:** Formulate the loss function based on binary classification.  
- **Implementation:** Compute binary cross-entropy loss and its gradient.  
- **Effect of Predictions:** Analyze how changing the model's output affects the loss.  

## **Notebook 5.3: Multiclass Cross-Entropy Loss**  
- **Categorical Distribution:** Understand how multi-class classification extends binary cross-entropy.  
- **Softmax Function:** Implement the softmax function for multi-class predictions.  
- **Negative Log-Likelihood:** Compute and visualize how the loss changes with predictions.  

## **Expected Outputs & Observations**  
- The loss functions should behave as expected when modifying inputs.  
- Cross-entropy loss should decrease when predictions match true labels.  
- The **Least Squares Loss** should exhibit a quadratic curve over error values.  

## **Contributing**  
If you find any errors or have suggestions for improvements, feel free to contribute by submitting a **pull request** or reporting an **issue**.  

## **License**  
This project is open-source and available under the **MIT License**.  

## **Code & Resources**  
📂 **Repository Link:** [Chapter-5]((https://github.com/SnehashisRatna/DeepLearning/tree/main/Notebooks/Chap05))  
📄 **Notebook Files:**  
- 📘 [Notebook 5.1: Least Squares Loss](https://github.com/SnehashisRatna/DeepLearning/blob/main/Notebooks/Chap05/5_1_Least_Squares_Loss.ipynb)  
- 📘 [Notebook 5.2: Binary Cross-Entropy Loss]((https://github.com/SnehashisRatna/DeepLearning/blob/main/Notebooks/Chap05/5_2_Binary_Cross_Entropy_Loss.ipynb))  
- 📘 [Notebook 5.3: Multiclass Cross-Entropy Loss]((https://github.com/SnehashisRatna/DeepLearning/blob/main/Notebooks/Chap05/5_3_Multiclass_Cross_entropy_Loss.ipynb))  




