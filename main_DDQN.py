# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

from MR_env import MR_env
from DMR_env import DMR_env
from DDQN import DDQN

#%%
env = MR_env(S_0=1, kappa=1, sigma=0.2, theta=1,
             dt=0.25, T = int(20), 
             I_max=10, lambd=0.05)
denv = DMR_env(S_0=1, kappa=1, sigma=0.2, theta_a=0.5, theta_b=1.5,
                dt=0.25, T = int(20), 
                I_max=10, lambd=0.05)

#%%
ddqn = DDQN(denv, I_max = 10,
            gamma = 0.999, 
            lr=1e-3,
            name="test" )
 
#%%    
ddqn.train(n_iter=10000, n_plot=200, n_iter_Q=5)