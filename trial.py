# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

from MR_env import MR_env
from DMR_env import DMR_env
from DDQN import DDQNAgent

#%%
env = MR_env(S_0=1, kappa=4, sigma=0.2, theta=1,
             dt=0.25, T = int(20), 
             I_max=10, lambd=0.05) 

# denv = MR_env(S_0=1, kappa=1, sigma=0.2, theta_a = 0.5, theta_b = 1.5,
#              dt=0.25, T = int(20), 
#              I_max=10, lambd=0.05)

ddqn = DDQNAgent(env, state_size=2, action_size=15, seed=0,
            I_max = 10,
            gamma = 0.999, 
            lr=1e-3,
            name="test" )

#%%    
ddqn.train(n_iter=1_000, n_plot=200, eps_start=1.0, eps_end=0.01, eps_decay=0.993)