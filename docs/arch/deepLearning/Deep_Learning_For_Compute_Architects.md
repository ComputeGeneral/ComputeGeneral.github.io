---
layout: default
---
[toc]

# 1 Introduction
> define key vocabulary, recap history and evolution of the techniques, and make the case for additional hardware support in the field.
## 1.1 the rises and falls of neural networks
![AI winter and AI history](./AI_History.png)

## 1.2 the third wave 
### 1.2.1 a virtuous cycle.
![virtuous Cycle](./virtuous_cycle.png)

## 1.3 the role of hardware in deep learning
### 1.3.1 state of the practice
- MNIST : a commonly used research dataset, 
![power consumption vs prediction Error](./powerConsumption_vs_predictionError.png)


---


# 2 Foundations of Deep Learning
> review basics of neural networks from linear regression to perceptrons and up to today's state-of-the-art deep neural networks    

## 2.1 Neural Networks
### 2.1.1 biological neural networks

### 2.1.2 Artificial neural networks
two categories of neurons research:
- models that replicate biological neurons to explain or understand their behavior (domain of biologist and cognitive scientists)
- solve arbitrary problems using neuron-inspired models of computation.(***neuromorphic computing***)


**preceptron** -- one of the earliest and still most widely used model of single artificial neuron: 
 
  $$y=\varphi(\sum_{i}^{}(w_i*x_i))$$

  $\varphi$: nonlinear **activation function**. vestiges to bilogical, loosely models the activation threshold of voltage-gated membrane proteins.
  $\sum_{i}^{}(w_i,x_i)$: a weighted sum of input $x_i$. vestiges to biological, reflects charge accumulation in the soma

<br>

**Neural Network Principle**: the ability of a neural net to model complex behavior is not due to sophisticated neurons, but to the aggregate behavior of many simple parts

**multilayer perceptron (MLP)** simplest neural network
  - *layer* (as the input to a neural network are often referred to as the "input layer", but it is not counted toward the overall *depth*)
  - *depth*: number of layer
  - *width*: number of neurons in a given layer.
  - *n*th layer of an MLP can be expressed as :    

  $$X_{n,i} = \varphi(\sum_{j}^{}(W_{n,i,j} * X_{n-1,j}))$$  
  
  or vector notation: 

  $$X_n = \varphi(W_n * X_{n-1})$$


  ![visualizing MLP](./visualizing_multilayer_perceptron.png)



**Activation Function**($\varphi$) 
a few nonlinear activation functions (historically):  
  - step-function, surperseded by continuous like: 
  - sigmoid
  - hyperbolic tangent
  - **Rectified Linear Unit (ReLU)**, modern neural nets tend to use.
    Definition of Widely used ReLU function is just the positive component of its input :    
  > why is it required?
  > why nonlinear? a well-known identity: composing any number of linear transformations produces another linear transformation. 
  >> i.e.: if $\varphi(x) = x$,    
  >> $x_1 = w_1*x_0$    
  >> $x_2 = w_2*x_1$    
  >> then we get $x_2 = w_2 * (w_1 * x_0)$    
  >> which also means: $x_2 = w' * x_0$    
  > it implies that without nonlinear activation function $\varphi$, a neural network of any depth is identical to some single-layer network.  


widely used ReLU function is just hte positive component of its input:
$$ReLU(x) = (x > 0) ? x : 0;$$

### 2.1.3 Deep Neural Networks (DNN)
Complexity fix: lower depth, needs wider width 
> expressing complicated functions (especially high dimensions) is prohibitively expensive for a two-layer MLP -- The hidden layer needs far too many neurons.

Instead, turn to networks with multiple narrower layers: DNN.
DNN's attractive in principle: 
*when solving a complicated problem, it is easier to break it into smaller pieces and then build upon those results.*
Advantages:
1. which is easier and faster for humans.
2. these smaller intermediate solutions can often be reused within the same neural network.

## 2.2 Learning
- **learn** - we use the word *learn* to describe the process of using these rules to adjust the parameters of a generic model such that it optimizes some objective.

- *Heart of deep learning*: we are not using the occult to capture some abstract concept; we are adjusting model parameters based on quantifiable metrics.

### 2.2.1 Types of Learning
- **supervised learning** (we have a set of matching inputs and outputs for some process or function and our goal is to predict the output of future inputs) which including two steps:
  - training phase: $x,y \rightarrow M$
  - inference phase: $M(x') \rightarrow y'$
- **unsupervised learning** (without sample output)
  - training phase: $x \rightarrow M$
  - inference phase: $M(x') \rightarrow y'$
  e.g. 1) clustering; 2)outlier detection
    (*generative models*)
- **reinforcement learning** (related to supervised learning, but decouples the form of the training output from that of the inference output.)
  - action: output of a reinforcement learning
  - reward: each training's input.
  - training phase: $x,r \rightarrow M$
  - inference phase: $M(x') \rightarrow y'$

###<font color=red>2.3.2 How Deep Neural Network Learn</font>
basic structure and characteristics: number of layers, size of layers and *activation function* are fixed; the values for neuron weight, by contrast, changed based on the data.

- **Loss functions**: one of the key design elements in training a neural network is what function we use to evaluate the <font color=red>*difference between the true and estimated outputs.* </font>
  - goal here is to find a function that will be minimized when the two are equal.
      - L1 norm of the difference: $L(y, y')=|y - y'|$
      - L2 norm, also know as: ***Root Mean Squared Error(RMSE)*** $L(y,y')=(y-y')^2$

  - cross-entropy loss:
    $$L(y,y') = -\frac{1}{n}\sum_{i}^{}ln(\frac{e^{y_i}}{\sum_{j}^{}e^{y'_j}})$$
    cross-entropy tends to be more effective for classification problems.

- **Optimization**: we want a way of adjusting the model weights to minimize our loss function.
  - *Stochastic Gradient Descent(SGD)*: *tells the direction in which to shift the estimate y'*
  - *Backpropagation*: *how to update the neural network to realize that shift(determiend by SGD)* the intuition behind backpropagation is that elements of a neural network should be adjust proportionally to the degree to which they contributed to generating the original output esitmate.


  gradient $\nabla{L(y,y')}$ is computing partial derivatives $\frac{\partial{L}}{\partial{x_i}}$ for all input $x$; for adjust individual weights: $\frac{\partial{L}}{\partial{w_{i,j}}}$, (every weight $j$, in every layer $i$) backpropagation is a mechanism for computing all of these partial loss components for every weight in a single pass.

  the DNN is *fully differentiable*, so the chain rule states that for multiple functions $f$, $g$, and $h$:
  $$\frac{\partial}{\partial{x}}f(g(h(x))) = \frac{\partial{f}}{\partial{g}} \frac{\partial{g}}{\partial{h}} \frac{\partial{h}}{\partial{x}}$$

  - *Cross-Entropy Loss math understanding:*
  <font color=blue><b>TODO</b></font>

<br>

- **Vanishing and Exploding Gradients**


---

# 3. Methods and Models
> dive into tools, workloads, and characterization. overview of modern neural network and machine learning software packages (TensorFlow, Torch, Keras, Theano and Caffe)

> present a collection of commonly used, seminal workloads that have assembled in a benchmark suit - Fathom. which will be break down into two categories: dataset(MNIST) and model.

## 3.1 An overview of advanced neural network methods

Aim: cheaper to compute, easier to train, robust to noise.

### 3.1.1 Model Architectures
- **Convolution Neural Networks(CNN)** a CNN works by sliding a large number of small trainable filters across an image.
  - mathematically: the forward operation is described by convolving an small filter with a large input image(hence the name). The backward (training) operation involves, naturally, deconvolving the gradients using the filters in order to produce a new set of gradients for the preceding layer.
  - An alternate interpretation of a CNN is as a large MLP where many of the weights in a layer are tied together in a specific pattern. In training, when a gradient update is applied to one of these weights, it is applied to all of them. These tied (or shared) weights correspond to a convolutional filter, and the structure with which they are connected describes the shape of the filter. In this interpretation, the computation of a CNN can be thought of as a compact representation of a larger MLP with identical mathematical properties

  AlexNet(2012)
  inception networks(2014.Google)
  Residual networks(2015.Microsoft)   

  <br>

- **Recurrent Neural Networks(RNN)** 
  suit with *input dependencies* such as speech and language. dependencies are relationships between inputs that change their meanings
  <br>
  RNNs are designed to deal with the sequential dependencies that often arise in time-series inputs. A RNN is a simple idea with complicated reprecussions: neurons use output as inputs in some way. 

  *unidirectional recurrent nets*
  *bidirectional networks*

  - Challenges when use RNN
    - 1. not easy to handle long-range dependencies.
    - 2. because of the inputs and outputs loopback the normal backpropagation algorithm no longer works.

    the first challenge motivated to develop a variety of recurrent neurons, the most popular one is *long-short-term memory* (LSTM)
<br>
    LSTM neurons have separate set of connections that propagate a signal to themselves and a set of "gates" that allow new values to be set or cleared from these connections. this act as a form of memory.
<br>
    LSTM are much more effective at capturing long-range dependencies.  
<br>
    *gated recurrent unit* a simplified version of LSTM has also proven popular.
  <br>

- **Other Model Architectures**
  - bipyramidal encoder/decoder model, which stacks two neural networks back to back (autoencoder)
  - read/write networks


### 3.1.2 Specialized layers
Modern Neural Networks utilize a variety of algorithmic tricks beyond perceptrons. **Most of** these tricks are described as layers, even though they may or may not contain anything that is trainable.  
- Pooling
- Normalization

## 3.2 Reference workloads for modern deep learning
### 3.2.1 Criteria For a Deep Learning Workload Suite
- Choose Meaningful Models: 

### 3.2.2 The Fathom workloads

## 3.3 Computational intuition behind deep learning
### 3.3.1 measurement and analysis in a deep learning framework.
### 3.3.2 Operation type profiling
### 3.3.3 Performance similarity
### 3.3.4 Training and inference
### 3.3.5 Parallelism and operation balance


# 4. Neural Network Accelerator Optimization: A Case Study
> Build off of Chapter3. review the Minerva accelerator design and optimization framework. include details of " how high-level neural network software libraries can be used in conglomeration with hardware CAD and simulation flows to co-design the algorithms and hardware.



# 5 A literature survey and review
> focus on the past decade and group papers based on the level in the compute stack (algorithmic, architecture, circuits) and by optimization type (sparsity, quantization, arithmetic approximation, and fault tolerance)
## 5.1 introduction
|abstraction   |algorithm|Architecture|Circuitt|
|:---          |:---     |:---        |:---    |
|dataType      | *       |   -        | -      |
|ModelSparsity | *       |   *        | -      |
|ModelSupport  | -       |   *        | -      |
|DataMovement  | -       |   *        | *      |
|FaultTolerance| -       |   -        | *      |



## 5.2 Algorithm
### 5.2.1 Data Type
### 5.2.2 Model Sparsity


# 6. Conclusion
> sheds light on areas that need attention and briefly outlines other areas of machine learning.