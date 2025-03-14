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

# **Shallow Neural Networks**  

## **Overview**  
This repository contains notebooks designed to help you understand and experiment with **shallow neural networks**, covering **1D and 2D inputs**, **activation functions**, and **network regions**.  

The key objectives of these notebooks are:  
- Understanding the architecture of shallow neural networks.  
- Experimenting with different **activation functions**.  
- Computing **loss functions** such as least squares error and negative log-likelihood.  
- Analyzing **network regions** and how activation functions impact them.  


## **Notebook 3.1: Shallow Neural Networks I**  
- **Understanding 1D Inputs:** Implement a shallow neural network with **one-dimensional input**.  
- **Activation Functions:** Experiment with different functions like **ReLU, Sigmoid, and Tanh**.  
- **Parameter Tuning:** Modify weights and biases to observe changes in model behavior.  
- **Loss Computation:** Compute **sum of squares loss and log-likelihood**.  

## **Notebook 3.2: Shallow Neural Networks II**  
- **Extending to 2D Inputs:** Implement a neural network that takes **two-dimensional inputs**.  
- **Gaussian Distribution:** Apply the Gaussian function and analyze probability distributions.  
- **Multiclass Classification:** Implement the **softmax function** and visualize classification behavior.  

## **Notebook 3.3: Shallow Network Regions**  
- **Computing Linear Regions:** Analyze the **maximum possible number of linear regions** a shallow network can form.  
- **Mathematical Derivation:** Understand theoretical justifications for network regions.  
- **Impact of Parameters:** Modify activation functions and network parameters to observe changes in decision boundaries.  

## **Notebook 3.4: Activation Functions**  
- **Exploring Activation Functions:** Compare different activation functions and their impact.  
- **Gradient Analysis:** Study the effect of different activation functions on gradient computation.  
- **Impact on Training:** Understand how activation choices affect convergence speed and loss minimization.  

## **Expected Outputs & Observations**  
- The model should fit **1D and 2D data points** efficiently.  
- Different activation functions will **change the network’s behavior** and predictions.  
- Softmax should correctly generate **probability distributions** over class labels.  
- The number of linear regions should increase as **network complexity grows**.  

## **📌 Code & Resources**  
- [Notebook 3.1 - Shallow Neural Networks I](https://github.com/SnehashisRatna/DeepLearning/blob/main/NoteBook(Chap-03)/3_1_Shallow_Networks_I.ipynb)  
- [Notebook 3.2 - Shallow Neural Networks II](https://github.com/SnehashisRatna/DeepLearning/blob/main/NoteBook(Chap-03)/3_2_Shallow_Networks_II.ipynb)
- [Notebook 3.3 - Shallow Network Regions](https://github.com/SnehashisRatna/DeepLearning/blob/main/NoteBook(Chap-03)/3_3_Shallow_Network_Regions.ipynb) 
- [Notebook 3.4 - Activation Functions](https://github.com/SnehashisRatna/DeepLearning/blob/main/NoteBook(Chap-03)/3_4_Activation_Functions.ipynb)  

## **Contributing**  
If you find any errors or have suggestions for improvements, feel free to contribute by submitting a **pull request** or reporting an **issue**.  

## **License**  
This project is open-source and available under the **MIT License**.  

## **Contact**  
For any queries, reach out to **udlbookmail@gmail.com**.    


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
📂 **Repository Link:** [Chapter-5](https://github.com/SnehashisRatna/DeepLearning/tree/main/Notebooks/Chap05)
📄 **Notebook Files:**  
- 📘 [Notebook 5.1: Least Squares Loss](https://github.com/SnehashisRatna/DeepLearning/blob/main/Notebooks/Chap05/5_1_Least_Squares_Loss.ipynb)  
- 📘 [Notebook 5.2: Binary Cross-Entropy Loss](https://github.com/SnehashisRatna/DeepLearning/blob/main/Notebooks/Chap05/5_2_Binary_Cross_Entropy_Loss.ipynb)
- 📘 [Notebook 5.3: Multiclass Cross-Entropy Loss](https://github.com/SnehashisRatna/DeepLearning/blob/main/Notebooks/Chap05/5_3_Multiclass_Cross_entropy_Loss.ipynb)




