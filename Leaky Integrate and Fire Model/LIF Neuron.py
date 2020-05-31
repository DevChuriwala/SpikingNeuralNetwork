__author__ = "Dev Churiwala"

"""
Plotting model, showcasing the Leaky Integrate-and-Fire Model
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m

def LIF_neuron(I_in):
  T            = 100                       # Total runtime (msec)
  dt           = 0.25                      # Timestep (msec)
  time         = np.arange(0, T+dt, dt)    # array in timesteps of dt
  Vm           = np.zeros(len(time))       # Potential u data(V)
  Rm           = 2                         # Resistor (kOhm)
  Cm           = 5                         # Capacitor (uF)
  tau_m        = Rm*Cm                     # Time Constant (msec)

  I            = np.zeros(len(time))
  I            = I_in

  Vth          = 1                         # Threshold (V)
  V_rest       = 0                         # Resting potential (V)
  Vm[0]        = V_rest 

  t_fire       = T + 2*dt                  # initialize firing time to beyond max time array value, as it was giving buggy runs
  t_refrac     = 5                         # refractory period

  for i in range(1,len(time)):
    Vm[i] = Vm[i-1] + ((-Vm[i-1]/tau_m)+((I[i-1]*Rm)/tau_m))*(dt)    # Euler Method

    if (time[i] > t_fire) and (time[i] < (t_fire+t_refrac)):
      Vm[i]  = V_rest                                                # set the V just after firing to V_rest as asked by assignment

    if (Vm[i] > Vth):
      Vm[i]  = V_rest                                                # current step is resting membrane potential
      t_fire = time[i]                                               # store firing time


  fig = plt.figure("Leaky Integrate and Fire Simulation", figsize=(14, 7))
  ax = fig.add_subplot(111)
  plt.title("Leaky Integrate and Fire Neuron Simulation")
  fig.subplots_adjust(left=0.1, bottom=0.32)
  line = plt.plot(time, Vm, label="Potential")[0]
  line2 = plt.plot(time, I, label="Applied Current")[0]
  plt.legend(loc="upper right")
  plt.ylabel("Potential [V]/ Current [A]")
  plt.xlabel("Time [s]")

  plt.show()
