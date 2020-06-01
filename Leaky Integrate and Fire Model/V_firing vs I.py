__author__ = "Dev Churiwala"

"""
Plotting model, showcasing the change in firing frequency as a function of constant current values
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
  fire_time    = []

  for i in range(1,len(time)):
    Vm[i] = Vm[i-1] + ((-Vm[i-1]/tau_m)+((I[i-1]*Rm)/tau_m))*(dt)    # Euler Method

    if (time[i] > t_fire) and (time[i] < (t_fire+t_refrac)):
      Vm[i]  = V_rest                                                # set the V just after firing to V_rest as asked by assignment

    if (Vm[i] > Vth):
      Vm[i]  = V_rest                                                # current step is resting membrane potential
      t_fire = time[i]                                               # store firing time
      fire_time.append(t_fire)

  if (len(fire_time) > 0) :
    frequency = round(1000/(fire_time[1] - fire_time[0]), 2)    
  else :
    frequency = 0  

  return frequency
  


I_test = np.arange(0, 10.025, 0.025)
V_firing = []
for i in range(0,len(I_test)):
  I_vary = np.zeros(len(time))
  I_vary.fill(I_test[i])
  V_firing.append(LIF_neuron(I_vary))

fig = plt.figure("V_firing vs I_test", figsize=(10, 7))
line = plt.plot(I_test, V_firing, color ="limegreen")[0]
plt.title("Firing Frequency vs. Current")
plt.ylabel("Firing Frequency [Hz]")
plt.xlabel("Current [A]")
plt.show()
