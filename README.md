# SAiDL Summer of Code

This repo contains the 2 models for the SSoC project. The first one is a Leaky Intgerate and Fire (LIF) model, which is modeled using numpy and matplotlib in Python3. The second one is a simple 2-layer shallow neural net, it is made to model the given input table for XOR/XNOR output of 2 binary bits using a third binary input bit.  

## Leaky Integrate and Fire Model

It is a basic Spiking Neuron model based on the following equations : <br />
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\tau\*\frac{du}{dt}&space;=&space;-&space;u&space;&plus;&space;IR" title="\tau\*\frac{du}{dt} = - u + IR" /> <br />
<img src="https://latex.codecogs.com/gif.latex?u(t^{(f)})=&space;\nu" title="u(t^{(f)})= \nu" /> <br />
<img src="https://latex.codecogs.com/gif.latex?\lim_{t&space;\rightarrow&space;t^{(f)},t&space;>&space;t^{(f)}}&space;=&space;u_{rest}" title="\lim_{t \rightarrow t^{(f)},t > t^{(f)}} = u_{rest}" /> <br />
<img src="https://latex.codecogs.com/gif.latex?u&space;=&space;u_{rest}\&space;if\&space;t^{(f)}<t<t^{(f)}&plus;t_{refrac}" title="u = u_{rest}\ if\ t^{(f)}<t<t^{(f)}+t_{refrac}" />
</p>

### Results

The observed results are plotted and saved as images in the Leaky Integrate and Fire Model folder.
The current used for each testcase is mentioned below.

```
Input Values :
I.   I = 2uA for 10ms <= t <= 75ms; I = 0Amp otherwise
II.  I = 1.5uA for 0ms <= t <= 100ms
III. I = 5sin(ωt)uA where ω = 45deg/msec
IV.  I = -2sin(ω1t) + 3sin(ω2t) + cos(ω3t)uA where ω1 = 45deg/msec, ω2 = 60 deg/msec, ω3 = 30 deg/msec
```

## Artificial Neural Network for XNOR/XOR

This is a basic model to predict the XOR/XNOR values of 2 binary single bit inputs based on a third bit (switch). It is a shallow network as the data is ordered and easy to classify.


### Input Values
| Bit 1 | Bit 2 | XNOR(0)/XOR(1) | Output |
| :---: | :---: | :---: | :---: |
| 1 | 1 | 0 | 1 |
| 1 | 0 | 0 | 0 |
| 0 | 1 | 0 | 0 |
| 0 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 |
| 1 | 0 | 1 | 1 |
| 0 | 1 | 1 | 1 |
| 0 | 0 | 1 | 0 |


## Author

**Dev Churiwala**
