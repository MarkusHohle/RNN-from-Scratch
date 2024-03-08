# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 03:02:45 2024

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt

X_t = np.arange(-10,10,0.1)
X_t = X_t.reshape(len(X_t),1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t),1)

plt.plot(X_t, Y_t)
plt.show()


from RNN import *

rnn = RunMyRNN(X_t,Y_t, Tanh(), n_epoch = 300)

X_new = np.arange(0,20,0.05)
X_new = X_new.reshape(len(X_new),1)

Y_hat = ApplyMyRNN(X_new,rnn)


plt.plot(X_t, Y_t)
plt.plot(X_new, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()

