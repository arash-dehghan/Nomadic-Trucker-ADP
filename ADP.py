from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
np.random.seed(0)

class Model:
	def __init__(self, V_matrix = {}, eps = 0.25):
		self.gamma = 0.9
		self.alpha = 0.05
		self.step = 0.01
		self.step_bar = 0.2
		self.lamb = 25
		self.l = 1
		self.locations = self.set_locations()
		self.distances = self.get_distances()
		self.initial_state = hash(1)
		self.demand = None
		self.V_matrix = V_matrix
		self.eps = eps
		self.beta = 0
		self.delta = 0
		self.sigma_s = 0

	def set_locations(self):
		coordinates = [(i,j) for j in np.linspace(0,1000,16) for i in np.linspace(0,1000,16)]
		return {location: {'c': coordinate, 'b': self.get_b(coordinate)} for location,coordinate in enumerate(coordinates,1)}

	def get_b(self, point):
		rescaled_x, rescaled_y = ((3.5 * point[0] / 1000) - 1.5), ((2 * point[1]/1000) - 1)
		return (1 - (self.get_f(rescaled_x, rescaled_y) + 1.03) / 6.03)

	def get_f(self, x, y):
		return min(4*(x**2) - 2.1*(x**4) + (1/3)*(x**6) + x*y - 4*(y**2) + 4*(y**4),5)

	def update_demands(self):
		ps = [self.locations[self.l]['b'] * (1 - self.locations[loc]['b']) for loc in self.locations.keys()]
		self.demand = {location: (1 if prob>= np.random.uniform(0,1) else 0) for location, prob in enumerate(ps,1)}

	def solve_system(self):
		z = self.get_C_costs() + (self.gamma * self.get_post_decision_costs())
		max_v = z[np.argmax(z)]
		return (max_v, np.random.randint(256)+1) if self.eps >= np.random.uniform(0,1) else (max_v, np.argmax(z)+1)

	def get_distances(self):
		return {(i,j): self.calculate_distance_points(i,j) for j in range(1,257) for i in range(1,257)}

	def calculate_distance_points(self, p1,p2):
		return np.linalg.norm(np.array(self.locations[p1]['c']) - np.array(self.locations[p2]['c']))

	def get_C_costs(self):
		return np.array([self.distances.get((self.l,location),0) * self.locations[self.l]['b'] if demand == 1 else (-1) * self.distances.get((self.l,location),0) for location, demand in self.demand.items()])

	def get_post_decision_costs(self):
		return np.array([self.V_matrix.get(hash(location), 0) for location in range(1,257)])

	def get_cost_of_action(self, a, o):
		return ((self.gamma**o)*self.distances.get((self.l,a),0) * self.locations[self.l]['b']) if self.demand[a] == 1 else ((-1) * (self.gamma**o)*self.distances.get((self.l,a),0))

	def update_stepsize(self,n,type,v_hat):
		if type == 'H':
			self.alpha = max(self.lamb / (self.lamb + n - 1), 0.05)
		elif type == 'B':
			self.step = self.step / (1 + self.step - self.step_bar)
			self.beta = (1 - self.step)*self.beta + self.step*(v_hat - self.V_matrix.get(hash(self.l), 0))
			self.delta = (1 - self.step)*self.delta + self.step*(v_hat - self.V_matrix.get(hash(self.l), 0))**2
			if n == 1:
				self.alpha = 1
				self.lamb = self.alpha**2
			else:
				self.sigma_s = (self.delta - (self.beta)**2) / (1 + self.lamb)
				self.alpha = 1 - (self.sigma_s / self.delta)
				self.lamb = ((1-self.alpha)**2)*self.lamb + (self.alpha)**2

	def update_V(self,v_hat, a_hat):
		if (1 - self.alpha) * self.V_matrix.get(hash(self.l), 0) + self.alpha * v_hat != 0:
			self.V_matrix[hash(self.l)] = (1 - self.alpha) * self.V_matrix.get(hash(self.l), 0) + self.alpha * v_hat

def side_simulation(m,O):
	total_reward = 0
	for o in range(O):
		m.update_demands()
		v_hat, a_hat = m.solve_system()
		total_reward += m.get_cost_of_action(a_hat,o)
		m.l = a_hat
	return total_reward

def myopic_result(K,O):
	results = []
	m = Model(V_matrix = {}, eps=-1)
	for k in range(K):
		results.append(side_simulation(m,O))
		m.l = 1
	return np.average(np.array(results))

def initial_state_simulation(update_stepsize=None, N=250, K=10, eps = 0.25):
	avg_initital_state_values = np.zeros(N+1)
	for k in tqdm(range(K)):
		values = np.zeros(N+1)
		m = Model(V_matrix = {}, eps=eps)
		for n in range(1,N+1):
			m.update_demands()
			v_hat, a_hat = m.solve_system()
			if update_stepsize is not None:
				m.update_stepsize(n,update_stepsize,v_hat)
			m.update_V(v_hat, a_hat)
			m.l = a_hat
			values[n] = m.V_matrix[m.initial_state]
		avg_initital_state_values += values
	return np.array(avg_initital_state_values) / K

def rewards_simulation(update_stepsize=None, N=250, K=10, M=10, O=10, eps = 0.25):
	M_runs = [i for i in range(0,N+M,M)]
	avg_total_rewards = np.zeros(len(M_runs)-1)
	side_m = Model(V_matrix = {}, eps=-1)
	for k in tqdm(range(K)):
		rewards = []
		m = Model(V_matrix = {}, eps=eps)
		for n in range(1,N+1):
			m.update_demands()
			v_hat, a_hat = m.solve_system()
			if update_stepsize is not None:
				m.update_stepsize(n,update_stepsize,v_hat)
			m.update_V(v_hat, a_hat)
			m.l = a_hat
			if n in M_runs:
				side_m.V_matrix = m.V_matrix
				rewards.append(side_simulation(side_m,O))
				side_m.l = 1
		avg_total_rewards += np.array(rewards)
	return avg_total_rewards / K




