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

# Chapter 6 – Optimization Algorithms

This chapter explores foundational optimization algorithms used to train neural networks. Each notebook builds intuition through interactive code, mathematical analysis, and visualizations. You'll be guided with **"TODO"** markers to write and experiment with code, promoting active learning.

---

## 📘 Notebook 6.1: Line Search

**Objective**:  
Understand how to find the minimum of a simple 1D function using line search.

**Key Concepts**:
- Numerical optimization
- Evaluating function values along a line
- Visualizing function curves and minima

**What You’ll Do**:
- Implement a basic line search algorithm
- Visualize function behavior
- Predict the location of minima based on curves

🔗 [View Notebook 6.1](notebooks/6.1_line_search.ipynb)

---

## 📘 Notebook 6.2: Gradient Descent

**Objective**:  
Recreate the classical **gradient descent** algorithm and understand how it updates parameters over time.

**Key Concepts**:
- Gradient computation
- Learning rate impact
- Loss landscape navigation

**What You’ll Do**:
- Implement gradient descent step-by-step
- Visualize descent paths on a 2D contour plot
- Explore how different learning rates affect convergence

🔗 [View Notebook 6.2](notebooks/6.2_gradient_descent.ipynb)

---

## 📘 Notebook 6.3: Stochastic Gradient Descent

**Objective**:  
Understand the differences between batch gradient descent and **stochastic gradient descent (SGD)**.

**Key Concepts**:
- Stochasticity in optimization
- Convergence vs. variance tradeoff
- Reproducing Figure 6.5

**What You’ll Do**:
- Implement SGD using a toy dataset
- Compare smooth vs. noisy trajectories
- Analyze learning stability

🔗 [View Notebook 6.3](notebooks/6.3_stochastic_gradient_descent.ipynb)

---

## 📘 Notebook 6.4: Momentum

**Objective**:  
Explore how **momentum** can help optimization escape poor local minima and accelerate convergence.

**Key Concepts**:
- Velocity-based updates
- Overdamped vs. underdamped behavior
- Momentum parameter tuning

**What You’ll Do**:
- Implement momentum manually
- Visualize how updates differ from standard gradient descent
- Recreate Figure 6.7 with momentum-enhanced updates

🔗 [View Notebook 6.4](notebooks/6.4_momentum.ipynb)

---

## 📘 Notebook 6.5: Adam Optimizer

**Objective**:  
Investigate the **Adam** optimization algorithm, which combines momentum and adaptive learning rates.

**Key Concepts**:
- Adaptive moment estimation
- Exponential moving averages
- Stability across diverse scenarios

**What You’ll Do**:
- Implement the Adam algorithm from scratch
- Compare it with SGD and momentum
- Reproduce Figure 6.9 to see how Adam converges

🔗 [View Notebook 6.5](notebooks/6.5_adam.ipynb)

---
# Chapter 7 – Gradients and Initialization

This chapter introduces the concept of computing gradients via **backpropagation** and discusses how **initialization** affects training dynamics in deep networks. The notebooks provide hands-on exercises to deepen understanding of how gradient signals flow and how smart initialization improves performance.

---

## 📘 Notebook 7.1: Backpropagation in Toy Model

**Objective**:  
Manually compute the derivatives of a simple toy model using chain rule principles, as discussed in Section 7.3.

**Key Concepts**:
- Function composition and derivatives
- Chain rule in depth
- Manual gradient tracking
- Least squares loss gradients

**What You’ll Do**:
- Analyze a model composed of known functions
- Calculate derivatives with respect to each parameter
- Understand how gradients flow through layers

🔗 [View Notebook 7.1](Chap07/7_1_Backpropagation_in_Toy_Model.ipynb)

---

## 📘 Notebook 7.2: Backpropagation

**Objective**:  
Implement the **backpropagation algorithm** on a deep neural network, as introduced in Section 7.4.

**Key Concepts**:
- Recursive gradient computation
- Layer-by-layer backward pass
- Intermediate variable tracking

**What You’ll Do**:
- Code the forward and backward passes of a neural net
- Track gradients at each layer
- Verify correctness using numerical gradient checking

🔗 [View Notebook 7.2](Chap07/7_2_Backpropagation.ipynb)

---

## 📘 Notebook 7.3: Initialization

**Objective**:  
Explore how different **weight initialization schemes** impact training, based on insights from Section 7.5.

**Key Concepts**:
- Variance scaling
- Vanishing and exploding gradients
- Xavier and He initialization

**What You’ll Do**:
- Experiment with various initialization strategies
- Observe their effect on signal propagation and loss curves
- Compare performance across architectures

🔗 [View Notebook 7.3](Chap07/7_3_Initialization.ipynb)

---
# Chapter 8 – Measuring Performance

Chapter 8 explores how we evaluate and understand the performance of neural networks. Key topics include performance on real-world datasets, the **bias-variance trade-off**, the **double descent** phenomenon, and the peculiar behavior of models in **high-dimensional spaces**.

---

## 📘 Notebook 8.1: MNIST_1D_Performance

**Objective**:  
Train and evaluate a simple neural network on the **MNIST-1D** dataset as shown in Figure 8.2a.

**Key Concepts**:
- Generalization
- Performance visualization
- Dataset preparation using `mnist1d`

**What You’ll Do**:
- Generate MNIST-1D data from [mnist1d repo](https://github.com/greydanus/mnist1d)
- Train a neural network model
- Visualize predictions and classification performance

🔗 [View Notebook 8.1](Chap08/8_1_MNIST_1D_Performance.ipynb)

---

## 📘 Notebook 8.2: Bias-Variance Trade-Off

**Objective**:  
Reproduce and understand the **bias-variance trade-off** as discussed in Section 8.3 and Figure 8.9.

**Key Concepts**:
- Underfitting vs. overfitting
- Model complexity
- Expected error decomposition

**What You’ll Do**:
- Fit models of varying complexity
- Measure training and test error
- Plot and analyze bias vs. variance curves

🔗 [View Notebook 8.2](Chap08/8_2_Bias_Variance_Trade_Off.ipynb)

---

## 📘 Notebook 8.3: Double Descent

**Objective**:  
Explore the **double descent** curve that appears when model capacity increases beyond the interpolation threshold.

**Key Concepts**:
- Classical vs. modern generalization
- Interpolation threshold
- Deep double descent behavior

**What You’ll Do**:
- Use `mnist1d` dataset
- Train networks of increasing size
- Plot risk curves and identify the double descent phenomenon

🔗 [View Notebook 8.3](Chap08/8_3_Double_Descent.ipynb)

---

## 📘 Notebook 8.4: High-Dimensional Spaces

**Objective**:  
Investigate unintuitive properties of high-dimensional spaces and their implications on machine learning models.

**Key Concepts**:
- Volume concentration
- Nearest neighbor distances
- Curse of dimensionality

**What You’ll Do**:
- Run simulations in high-dimensional space
- Visualize distance distributions
- Understand why high dimensions pose challenges for ML models

🔗 [View Notebook 8.4](Chap08/8_4_High_Dimensional_Spaces.ipynb)

---
# Chapter 10 – Convolutional Networks

This chapter introduces convolutional neural networks (CNNs), which are foundational for working with image and spatial data. These notebooks help you understand **1D and 2D convolution**, how to implement them from scratch, and how to use them in real-world datasets like **MNIST**.

---

## 📘 Notebook 10.1: 1D Convolution

**Objective**:  
Understand and implement **1D convolutional layers**, focusing on how filters interact with input signals.

**Key Concepts**:
- Convolution vs. correlation
- Filter stride, padding, and dilation
- Receptive fields in 1D data

**Important Note**:  
This notebook corrects a **notation issue in the printed book**—it follows the standard definition of dilation where:
- Dilation 1 = no space between filter elements  
- Dilation 2 = one space between filter elements  
Refer to the [errata](https://udlbook.github.io/udlbook/errata.html) if needed.

🔗 [View Notebook 10.1](Chap10/10_1_1D_Convolution.ipynb)

---

## 📘 Notebook 10.2: Convolution for MNIST-1D

**Objective**:  
Build and train a 1D convolutional neural network on the **MNIST-1D dataset**, as seen in Figures 10.7 and 10.8a.

**Key Concepts**:
- Feature extraction with convolutions
- Pooling and downsampling in 1D
- Performance comparison to dense models

**What You’ll Do**:
- Implement a full CNN pipeline for 1D data
- Evaluate model performance

🔗 [View Notebook 10.2](Chap10/10_2_Convolution_for_MNIST_1D.ipynb)

---

## 📘 Notebook 10.3: 2D Convolution

**Objective**:  
Understand the core **2D convolution operation** by implementing it manually and comparing it with PyTorch's output.

**Key Concepts**:
- Kernel sliding and summation
- Cross-checking manual vs. library results
- Stride, padding, and filter size impact

**What You’ll Do**:
- Manually implement 2D convolution
- Validate correctness using PyTorch

🔗 [View Notebook 10.3](Chap10/10_3_2D_Convolution.ipynb)

---

## 📘 Notebook 10.4: Downsampling and Upsampling

**Objective**:  
Explore how **downsampling** (e.g., max pooling) and **upsampling** (e.g., interpolation) affect signal resolution in CNNs.

**Key Concepts**:
- Subsampling for translation invariance
- Different upsampling strategies
- Information loss and recovery

**What You’ll Do**:
- Experiment with resolution changes
- Analyze effects on feature maps

🔗 [View Notebook 10.4](Chap10/10_4_Downsampling_and_Upsampling.ipynb)

---

## 📘 Notebook 10.5: Convolution for MNIST (2D)

**Objective**:  
Build a full 2D CNN using the classic **MNIST** handwritten digits dataset.

**Key Concepts**:
- Real-world application of CNNs
- MNIST 28x28 input handling
- Classification into 10 digit classes

**Reference**:  
Adapted from [this PyTorch MNIST example](https://nextjournal.com/gkoehler/pytorch-mnist).

🔗 [View Notebook 10.5](Chap10/10_5_Convolution_For_MNIST.ipynb)

---

📬 **Contact**  
If you find any mistakes or have suggestions, feel free to reach out:  
📧 **udlbookmail@gmail.com**









