# SNN Model

## i) Framing a Question

As time progresses our Neural Networks are becoming deeper and there is tons of data to process. Conventionally, computing this via ANNs consumes a lot of energy, which is often not possible on edge//IOT devices. Hence, we seek to find a way of creating an energy-efficient and accurate model to classify images. The ideal starting point to base our model off of is the human brain which consumes only 12 watts - minuscule compared to the GPUs we use to train ANNs. So, two questions arise,(1) Can we mimic the brain's efficiency in our model whilst maintaining the accuracy of conventional models? And (2) How can we integrate the best ideas from previously published work on SNNs to optimize our model and reduce the existing gap between ANNs and SNNs? To answer both the questions, we first need to understand the current State of the art (SOTA) in the field of SNNs.

## ii) Understanding the SOTA

<img src="./Images/SNN_Class.png" width="800">
As shown above, the SOTA models for SNNs currently consist of (a)Converted, (b)Unsupervised, and (c)Supervised SNNs.<br>
(a) Converted SNNs use advanced ANN optimization techniques and boast nearly ANN level accuracies(Rueckauer et al., 2017; Sengupta et al., 2019) but the problem arises in the signal representation, wherein the frequencies need to be estimated requiring non-trivial passage on time,<br>
(b) Unsupervised methods such as STDP are biologically plausible and show high accuracies on datasets such as MNIST(Diehl and Cook, 2015) but the method does not scale well to deeper networks and hence has limited expressivity,<br>
(c) Supervised methods of training on Spike-Based Data requires non-conventional BP, but it shows promise in terms of energy efficiency and better generalization, thus achieving better accuracies.<br>

We also look at previously implemented models and their results, which can be found in the Images Folder. SNN_Results(Lee et al., 2020) shows the best models on popular datasets, with accuracies on CIFAR10 reaching 90% using Spike-based BP. SNN_Results_2(Diehl and Cook, 2015) depicts the results of past works on the MNIST dataset and SNN_Results_3(Sengupta et al., 2019) depicts the results of past BP-based works on the CIFAR10 dataset. These results leave us with a fair idea of how our model should perform to be competitive compared to already established results.

## iii) Basic Ingredients

We are interested in classifying Images, hence the input to the model will be images encoded as Poisson-distributed Spike trains, wherein the pixel intensity dictates the probability of spike generation. We will use the spike-based gradient descent algorithm as derived by Lee et al., 2020 for the Leaky Integrate and Fire(LIF) Neuron. We will also incorporate the ideas from successful deep ANN models such as LeNet, VGG, and EfficientNet. We plan to adapt Dropout and BatchNorm (non-conventional for spiking data) techniques to better regularize the network. We aim to benchmark the results against past work on object datasets such as MNIST and CIFAR10.<br />
The code for modeling the LIF Neuron has already been implemented by me, and can be found at, <https://github.com/DevChuriwala/SpikingNeuralNetwork>.

## iv) Our Hypothesis

We hypothesize that the incorporation of Dropout and BatchNorm(Ledinauskas et al., 2020) should improve the regularization performance of our model over the existing work, the latter also helping in reducing the training time. Since we are using supervised learning, we also hypothesize that using deeper networks, we should be able to create a model with performance comparable to leading ANNs.<br>

(a) LIF Neuron Model:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\tau\*\frac{du}{dt}&space;=&space;-&space;u&space;&plus;&space;IR" title="\tau\*\frac{du}{dt} = - u + IR" /> <br />
<img src="https://latex.codecogs.com/gif.latex?\bg_white&space;u(t^{(f)})=&space;\nu" title="u(t^{(f)})= \nu" /> <br />
<img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\lim_{t&space;\rightarrow&space;t^{(f)},t&space;>&space;t^{(f)}}&space;=&space;u_{rest}" title="\lim_{t \rightarrow t^{(f)},t > t^{(f)}} = u_{rest}" /> <br />
<img src="https://latex.codecogs.com/gif.latex?\bg_white&space;u&space;=&space;u_{rest}\&space;if\&space;t^{(f)}<t<t^{(f)}&plus;t_{refrac}" title="u = u_{rest}\ if\ t^{(f)}<t<t^{(f)}+t_{refrac}" />
</p>

(b) Modeling Spike-based BP:
<p align="center">
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\frac{\partial&space;E}{\partial&space;w^{L}}&space;=&space;\frac{\partial&space;E}{\partial&space;a_{LIF}}&space;\frac{\partial&space;a_{LIF}}{\partial&space;net}&space;\frac{\partial&space;net}{\partial&space;w^{L}}" title="\frac{\partial E}{\partial w^{L}} = \frac{\partial E}{\partial a_{LIF}} \frac{\partial a_{LIF}}{\partial net} \frac{\partial net}{\partial w^{L}}" /><br/>
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;Final\&space;output\&space;error,\&space;e_{j}&space;=&space;output_{j}&space;-&space;label_{j}" title="Final\ output\ error,\ e_{j} = output_{j} - label_{j}" /><br/>
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;Loss\&space;Function,\&space;E&space;=&space;\frac{1}{2}\sum_{j=1}^{n^{L}}e_{j}^{2}" title="Loss\ Function,\ E = \frac{1}{2}\sum_{j=1}^{n^{L}}e_{j}^{2}" /><br/>
</p>

## v) Toolkits

Having worked extensively with Python, we decided to use it as our language of choice, it is relatively easy to code in and the abundance of useful libraries is very helpful. As for SNN specific libraries, there are many to choose from such as BRIAN, ANNarchy, NEST, NEURON, Nengo(SpiNNaker) and BindsNET; but our goal is to make an efficient model which runs on CPU/GPUs, and having had a lot of experience using PyTorch in the past, we chose BindsNET(Hazan et al., 2018). We carry out most of our model training on Colab - generally on the Tesla P100 GPUs and the datasets we use for benchmarking are research standards such as MNIST and CIFAR10.

## vi) Plan 

The below images are taken from Lee et al., 2020, graphically explain our plan and the base for our model to develop on.

<img src="./Images/LIF_Rep.png" width="800"><br/>
The above image shows the input image encoded as Poisson-distributed spike trains entering layer L, where they are passed through the LIF neuron as described above, and output as a spike train to be passed into layer L+1.

<img src="./Images/SNN_Conv.png" width="800"><br/>
The above image shows the working of the convolutional and spatial pooling layer. For each timestep dt, the spikes are convolved with the weights, get accumulated over time and continuously leak. When the accumulated voltage becomes higher than the threshold voltage, a spike is generated.

<img src="./Images/SNN_Arch.png" width="800"><br/>
A figure showing the forward and backward propagation steps of our model, similar to what we see in ANN propagation.

## ix) Testing the model

Initially, we create a baseline model without Dropout, BatchNorm, or Data Augmentation to get a fair measure of its capabilities. We add a manual seed to allow all the results to be reproducible. The dataset is divided into a fixed training and validation set using the seed, with the validation set being used only for testing purposes and is unseen to the model during training. Since our model is an image classifier with equivalent class sizes, accuracy is a good measure of model performance. We selectively add features and test the model accuracy to see their individual and collective effects on the same seeded validation data(we also plan to explore K-Fold CV for the same). We hypothesize an increase in model performance with an increase in model depth, so we will also be looking at the change in precision versus the model size. Ideally, we aim to bridge the gap between ANNs and SNNs, so we numerically compare ANN accuracies to our model's accuracies(in terms of error increment) on the same data.



### PS 
This is an ongoing project so we are still in the process of (vii)implementing the model. After (viii)completion of the model, we will move on to (ix)evaluation and optimization.
Code for the project will periodically be put up in my SNN repo, which can be found at, <https://github.com/DevChuriwala/SpikingNeuralNetwork>. 
