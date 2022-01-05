from matplotlib import pyplot as plt
import numpy as np

def run_initial_state(N=25000, K=10):
	Exp_H = initial_state_simulation(update_stepsize='H', N=N, K=K, eps=0)
	Exp_B = initial_state_simulation(update_stepsize='B', N=N, K=K, eps=0)
	Eps025_H = initial_state_simulation(update_stepsize='H', N=N, K=K, eps=0.25)
	Eps025_B = initial_state_simulation(update_stepsize='B', N=N, K=K, eps=0.25)
	Eps1_H = initial_state_simulation(update_stepsize='H', N=N, K=K, eps=1)
	Eps1_B = initial_state_simulation(update_stepsize='B', N=N, K=K, eps=1)

	plt.plot([i for i in range(N+1)], Exp_H, linestyle = 'dashdot', color = 'b', label='Exp-H')
	plt.plot([i for i in range(N+1)], Eps025_H, linestyle = 'dotted', color = 'b', label='Eps025-H')
	plt.plot([i for i in range(N+1)], Eps1_H, linestyle = 'dashed', color = 'b', label='Eps1-H')
	plt.plot([i for i in range(N+1)], Exp_B, linestyle = 'dashdot', color = 'g', label='Exp-B')
	plt.plot([i for i in range(N+1)], Eps025_B, linestyle = 'dotted', color = 'g', label='Eps025-B')
	plt.plot([i for i in range(N+1)], Eps1_B, linestyle = 'dashed', color = 'g', label='Eps1-B')
	plt.xlabel('Iteration (n)')
	plt.ylabel('Estimated Value of State 1')
	plt.title(f'Average Estimated Value of State 1 (K={K} Iterations)')
	plt.legend()
	plt.savefig('Initial_State_Values.png')

def run_rewards(N=25000, K=10, M=10, O=30):
	myopic = myopic_result(K,O) 
	Exp_H = rewards_simulation(update_stepsize='H', N=N, K=K, M=M, O=O,  eps=0)
	Exp_B = rewards_simulation(update_stepsize='B', N=N, K=K, M=M, O=O,  eps=0)
	Eps025_H = rewards_simulation(update_stepsize='H', N=N, K=K, M=M, O=O, eps=0.25)
	Eps025_B = rewards_simulation(update_stepsize='B', N=N, K=K, M=M, O=O, eps=0.25)
	Eps1_H = rewards_simulation(update_stepsize='H', N=N, K=K, M=M, O=O, eps=1)
	Eps1_B = rewards_simulation(update_stepsize='B', N=N, K=K, M=M, O=O, eps=1)

	plt.plot([i for i in range(0,N+M,M)], np.insert(Exp_H, 0, myopic), linestyle = 'dashdot', color = 'b', label='Exp-H')
	plt.plot([i for i in range(0,N+M,M)], np.insert(Eps025_H, 0, myopic), linestyle = 'dotted', color = 'b', label='Eps025-H')
	plt.plot([i for i in range(0,N+M,M)], np.insert(Eps1_H, 0, myopic), linestyle = 'dashed', color = 'b', label='Eps1-H')
	plt.plot([i for i in range(0,N+M,M)], np.insert(Exp_B, 0, myopic), linestyle = 'dashdot', color = 'g', label='Exp-B')
	plt.plot([i for i in range(0,N+M,M)], np.insert(Eps025_B, 0, myopic), linestyle = 'dotted', color = 'g', label='Eps025-B')
	plt.plot([i for i in range(0,N+M,M)], np.insert(Eps1_B, 0, myopic), linestyle = 'dashed', color = 'g', label='Eps1-B')
	plt.xlabel('Iteration (n)')
	plt.ylabel('Average Discounted Rewards')
	plt.title(f'Discounted Rewards Given N={N}, M={M}, K={K}, O={O}')
	plt.legend()
	plt.savefig('Discounted_Rewards.png')







