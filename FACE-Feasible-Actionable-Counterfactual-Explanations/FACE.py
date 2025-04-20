import numpy as np
import pandas as pd
import pickle as pk

import utils
from kernel import *

random_seed = 482
np.random.seed(random_seed)

class FACE:
	def __init__(self, data, density_method, distance_obj, kernel_obj, feature_columns, target_column, epsilon, clf=None):
		self._data = data
		self._density = density_method
		self._distance = distance_obj
		self._epsilon = epsilon
		self._clf = clf
		self._kernel_obj = kernel_obj
		self.set_feature_names(feature_columns, target_column)

		self._Graph = None

	def set_feature_names(self, features, target):
		self.FEATURE_COLUMNS = features
		self.TARGET_COLUMN = target

	def getDistance(self, xi, xj):
		self._distance.computeDistance(xi, xj)

	def train_classifier(self):	
		if (self._clf is None):
			self._clf = LogisticRegression(random_state=random_seed)

		X = self._data[self.FEATURE_COLUMNS]
		y = self._data[self.TARGET_COLUMN]
		self._clf.fit(X, y)
		print("Trained classifier")

	def shortestPath(self, source, target):
		"""
		Djikstra to find the shortest path in Graph
		source: source_id
		target: target_id
		Returns shortest_path, path_cost
		"""
		path = [source]
		path_cost = 0
		current = source
		visited = []
		if not (current in self._Graph):
				#print(f"Source node {source} not in graph!")
				return [], -1
		## while current is not target
		while (current is not target):
			## update distances of all neighbors
			visited.append(current)
			if not (current in self._Graph):
				#print(f"Node {current} has no neighbors.")
				return [], -1

			distances = self._Graph[current]
			#print(distances)
			minimum_cost = float('inf')
			closest = -1

			## Pick the closest neighbor
			#print(f"Voisins de {current} ({current == 22}): {distances}")
			for key in distances:
				#print(key)
				if key in visited:
					#print("continue ???")
					continue
				#print("continue")
				#try:
					#weight = float(distances[key])
					#print(weight)
				#except Exception as e:
					#print(f"Erreur de lecture du poids de {current} vers {key} : {distances[key]}")
					#continue
				#print(f"Checking node {key}, weight: {weight}, current min cost: {minimum_cost}")
				weight = float(distances[key])
				#if weight < minimum_cost:
					#minimum_cost = weight
					#print(minimum_cost)
					#closest = key
					#print(key)
				#if key not in visited:
					
					#print(f"Checking node {key}, weight: {weight}, current min cost: {minimum_cost}")
				if ((weight < minimum_cost) and (key not in visited)):
					minimum_cost = weight
					#print(minimum_cost)
					closest = key
					#print(closest)
			#print("--------------------------------------")
			if (closest == -1):
				#print(f"No closest node found from node {current}. Returning empty path.")
				break
				#return [], -1

			path.append(closest)
			#print(f"path: {path}")
			path_cost += float(distances[closest])
			#print(f"path_cost: {path_cost}")
			current = closest
			#print(f"current: {current}")

		#print(f"Found path: {path}, with cost: {path_cost}")
		return path, path_cost
	
	def plus_court_chemin_recursive(graph, current_node, target_node, visited=None, path=None, cost=0):
		if visited is None:
			visited = set()
		if path is None:
			path = []
		visited.add(current_node)
		path = path + [current_node]

		if current_node == target_node:
			return path, cost
		
		min_path = None
		min_cost = float('inf')
		for neighbor, weight_array in graph[current_node].items():
			if neighbor not in visited:
				weight = weight_array[0] if isinstance(weight_array, (list, np.ndarray)) else weight_array
				new_path, new_cost = plus_court_chemin_recursive(graph, neighbor, target_node, visited.copy(), path, cost + weight)
				if new_path is not None and new_cost < min_cost:
					min_path = new_path
					min_cost = new_cost
		return min_path, min_cost if min_path is not None else (None, float('inf'))

	def make_graph(self, feasibilitySet, epsilon):
		"""
		Make the graph using given data points
		"""
		print("Constructing graph...")
		X = self._data[self.FEATURE_COLUMNS]
		# self._Graph = utils.make_graph(X, self._density, self._distance, self._kernel_obj, feasibilitySet, epsilon) ## Adjacency matrix representation of graph
		self._Graph = utils.make_graph_adjList(X, self._density, self._distance, self._kernel_obj, feasibilitySet, epsilon) ## Adjacency matrix representation of graph
		pk.dump(self._Graph, open("Graph_faceSynthetic.pk", 'wb'))

	def get_candidates(self, tp, td):
		"""
		Returns a dictionary of idices of candidate points for recourse
		"""
		candidates = {}
		for x_id, x in self._data[self.FEATURE_COLUMNS].iterrows():
			# print("pdf", self._density.pdf(x, self._data.iloc[x_id][self.TARGET_COLUMN]))
			if (self._clf.predict_log_proba([x])[0][1] > np.log(tp)): # and self._density.pdf(x, self._data.iloc[x_id][self.TARGET_COLUMN]) > td):
				candidates[x_id] = x

		return candidates

	def compute_recourse(self, source, tp, td):
		"""	
		source: Source point id
		X: dataset - pd dataframe
		density: probability density function
		distance: distance function between 2 points
		tp: threshold for classifier prediction probability
		td: threshold for pdf
		epislon: constant for distance threshold
		c: constrains for feasibility
		clf: classifier sklearn object
		"""		
		assert (self._Graph is not None)

		I = self.get_candidates(tp, td) ### Indices of candidates
		#print(f"{len(I)} candidats trouvés pour le recourse")
		#print(type(I))
		#print(I)
		min_path_cost = float('inf')
		min_target = -1
		min_path = None
		for candidate_id in I:
			#print(candidate_id)
			candidate = I[candidate_id]			
			closest_target_path, path_cost = self.shortestPath(source, candidate_id)
			#print(f"path_cost:{path_cost}")
			#print(f"closest target: {closest_target_path}")
			if (path_cost == -1):
				continue
			if (path_cost < min_path_cost):
				min_target = candidate
				min_path_cost = path_cost
				min_path = closest_target_path

		return min_target, min_path_cost, min_path

	#### Unit tests
	def unit_test_djikstra(self):
		G = {0: {1: 3, 2:4}, 1: {0: 3, 2: 4}, 2: {0:4, 1: 4, 3: 3}, 3: {2: 3, 4: 4, 5: 1}, 4:{3: 4, 5: 1}, 5:{3:1, 4:1}}
		self._Graph = G
		path, path_cost = self.shortestPath(0, 5)
		print("Path:", path, " Cost:", path_cost)
		assert(path == [0, 1, 2, 3, 5])
		# assert(path_cost == 7)
		print("unit test djikstra passed...")

