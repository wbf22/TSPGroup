#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	class cityData:
		def __init__(self, prev, cit):
			self.previous = prev
			self.city = cit


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	# startingCity = None
	# finalCity = None
	# def greedy(self, time_allowance=60.0):
	# 	start_time = time.time()
	# 	# initialize unvisited[] to all cities
	# 	unvisited = self._scenario.getCities().copy()  # would it be better to just have an array of
	# 	# sizeof(unvisited) and initialize all values to 0?
	# 	# previousPointers = []
	# 	# for c in unvisited:
	# 	# 	previousPointers.append(self.cityData(None, c))
	#
	# 	startingCityIndex = 0
	# 	startingCity = unvisited[startingCityIndex]
	#
	# 	route = []
	# 	currentCity = startingCity
	# 	while len(unvisited) != 0:
	# 		unvisited.remove(currentCity)
	# 		route.append(currentCity)
	# 		# find shortest path to city2 from current city1
	# 		nextCity = self.findClosestCity(currentCity, unvisited)
	#
	# 		# update previous pointers
	# 		# previousPointers[nextCity._index].previous = currentCity
	# 		if len(unvisited) == 0:
	# 			if self.checkLastCity(startingCity, currentCity) == False:
	# 				startingCityIndex += 1
	# 				unvisited = self._scenario.getCities().copy()
	# 				route.clear()
	# 				currentCity = unvisited[startingCityIndex]
	# 				nextCity = currentCity
	# 				continue
	# 			else: break
	#
	# 		if nextCity == currentCity:
	# 			startingCityIndex += 1
	# 			unvisited = self._scenario.getCities().copy()
	# 			route.clear()
	# 			currentCity = unvisited[startingCityIndex]
	#
	# 		currentCity = nextCity
	#
	# 	results = {}
	# 	foundTour = False
	# 	bssf = TSPSolution(route)
	# 	count = len(route)
	# 	if bssf.cost < np.inf:
	# 		# Found a valid route
	# 		foundTour = True
	# 	end_time = time.time()
	# 	results['cost'] = bssf.cost if foundTour else 999999
	# 	results['time'] = end_time - start_time
	# 	results['count'] = count
	# 	results['soln'] = bssf
	# 	results['max'] = None
	# 	results['total'] = None
	# 	results['pruned'] = None
	#
	# 	return results
	#
	# def findClosestCity(self, currentCity, unvisited):
	#
	# 	# loop through all edges of city1
	# 	edges = self._scenario.getEdges()[currentCity._index]
	# 	cities = self._scenario.getCities()
	# 	shortestDistance = 999999
	# 	closestCity = currentCity
	# 	for i in range(len(edges)):
	# 		if edges[i] and (cities[i] in unvisited ):
	# 			distance = self.calculateDistance(cities[i], currentCity)
	# 			if distance < shortestDistance:
	# 				shortestDistance = distance
	# 				closestCity = cities[i]
	#
	# 	return closestCity
	#
	#
	# def checkLastCity(self, firstCity, currentCity):
	# 	edges = self._scenario.getEdges()[currentCity._index]
	# 	if edges[firstCity._index] == True:
	# 		return True
	#
	# 	return False
	#
	#
	# def calculateDistance(self, city1, city2):
	# 	y = city1._y - city2._y
	# 	x = city1._x - city2._x
	#
	# 	dist = np.sqrt(y**2 + x**2)
	#
	# 	return dist
	startingCity = None
	finalCity = None

	def greedy(self, time_allowance=60.0):

		start_time = time.time()
		foundTour = False
		startingCityIndex = 0
		self.cities = self._scenario.getCities()
		results = {}
		while foundTour == False:

			unvisited = self._scenario.getCities().copy()  # would it be better to just have an array of

			startingCity = unvisited[startingCityIndex]

			# keep looping until all nodes are visited, will reset if path isn't complete
			route = []
			currentCity = startingCity
			while len(unvisited) != 0:
				unvisited.remove(currentCity)
				route.append(currentCity)
				# find shortest path to city2 from current city1
				nextCity = self.findClosestCity(currentCity, unvisited)

				if len(unvisited) == 0:
					# reset if failed at end of tour
					if self.checkLastCity(startingCity, currentCity) == False:
						startingCityIndex += 1
						unvisited = self._scenario.getCities().copy()
						route.clear()
						currentCity = unvisited[startingCityIndex]
						nextCity = currentCity
						continue
					else:  # break if successful
						break

				# reset if failed attempt midway through tour
				if nextCity == currentCity:
					startingCityIndex += 1
					unvisited = self._scenario.getCities().copy()
					route.clear()
					if startingCityIndex >= len(unvisited): break
					currentCity = unvisited[startingCityIndex]

				currentCity = nextCity

			# if no tour was found return default random tour
			if startingCityIndex == len(self._scenario.getCities()): return self.defaultRandomTour(time_allowance)
			bssf = TSPSolution(route)
			count = len(route)
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else 999999
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results

	# helps find next city to visit using
	def findClosestCity(self, currentCity, unvisited):

		# loop through all edges of city1
		edges = self._scenario.getEdges()[currentCity._index]
		cities = self._scenario.getCities()
		shortestDistance = 999999
		closestCity = currentCity
		for i in range(len(edges)):
			if edges[i] and (cities[i] in unvisited):
				distance = self.calculateDistance(cities[i], currentCity)
				if distance < shortestDistance:
					shortestDistance = distance
					closestCity = cities[i]

		return closestCity

	def checkLastCity(self, firstCity, currentCity):
		edges = self._scenario.getEdges()[currentCity._index]
		if edges[firstCity._index] == True:
			return True

		return False

	def calculateDistance(self, city1, city2):
		# y = city1._y - city2._y
		# x = city1._x - city2._x
		#
		# dist = np.sqrt(y**2 + x**2) * 1000

		return city1.costTo(city2)
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		pass



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		start_time = time.time()
		#pick arbitrary node
		#find node outside C closest to a node in C called K
		#find edge in C that dik+dkj-dij is minimal
		#Make new cycle replacing ij with ik and kj
		#keeping going until C has all vertices
		self.nodes = self._scenario.getCities()
		self.edges = self._scenario.getEdges()
		self.path = [self.nodes[0]]
		self.nodes.remove(self.nodes[0])

		while len(self.nodes) > 0:
			self.addNode()

		results = {}
		foundTour = False
		bssf = TSPSolution(self.path)
		count = len(self.path)
		if bssf.cost < np.inf:
			# Found a valid route
			foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else 999999
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results

	def addNode(self):
		#using the "5: Random Insertion" algorithm from the website https://stemlounge.com/animated-algorithms-for-the-traveling-salesman-problem/
		index = -1
		#keep going until you find a valid node to add
		while index == -1:
			newNode = self.nextNodeToInsert()
			index = self.whereToInsertNode(newNode)

		self.path.insert(index, newNode)
		self.nodes.remove(newNode)

	def whereToInsertNode(self, newNode):
		#deals with problem of when you only have one node in path
		if len(self.path) == 1:
			if self.edges[self.path[0]._index][newNode._index] == True:
				return 1
			else:
				return -1
		else: #once you have more than one node in path

			#find pair of nodes in path that can connect to newNode
			possibleEdgesToReplace = []
			for n in range(len(self.path)):
				if self.edges[self.path[n]._index][newNode._index] == True:
					if n == len(self.path) - 1:
						if self.edges[newNode._index][self.path[0]._index] == True:
							possibleEdgesToReplace.append((n, 0))
					else:
						if self.edges[newNode._index][self.path[n + 1]._index] == True:
							possibleEdgesToReplace.append((n, n + 1))
			if len(possibleEdgesToReplace) == 0: return -1

			#of those pairs that can connect, choose the best one
			bestPair = None
			bestDist = math.inf
			for pair in possibleEdgesToReplace:
				dist = self.calculateDistance(self.path[pair[0]], newNode) + self.calculateDistance(self.path[pair[1]], newNode)
				if bestDist > dist:
					bestPair = pair
					bestDist = dist

			if bestPair == None:
				return -1

			return bestPair[1]

	def nextNodeToInsert(self):


		closestNode = None
		while closestNode == None:
			node = random.choice(self.path)
			for n in self.nodes:
				if closestNode == None:
					if self.edges[node._index][n._index] == True:
						closestNode = n
				elif node.costTo(n) < node.costTo(closestNode):
					closestNode = n


		return closestNode


	# ###!!! I had a problem when you want to add the closest node but none of the nodes in the path have edges to it
	#these are the functions we made thursday april 1st
	# def addNode(self):
	# 	closestNode = self.getClosestNode()
	# 	self.whereToInsertNode(closestNode)
	#
	# 	pass
	#
	# def getClosestNode(self):
	# 	chosenNode = random.choice(self.path)
	# 	closetNode = self.nodes[0]
	# 	for n in self.nodes:
	# 		if self.calculateDistance(n, chosenNode) < self.calculateDistance(closetNode, chosenNode):
	# 			closetNode = n
	#
	# 	return chosenNode
	#
	# ###!!! I had a problem when you want to add the closest node but none of the nodes in the path have edges to it
	# def calcNodesClosestToFarthest(self):
	#
	#
	# def whereToInsertNode(self, node):
	# 	i = None
	# 	j = None
	#
	# 	#check placing each node in between each pair in the current path
	# 	for n in range(self.path):
	# 		#see if any node in path connects to
	# 		if self.edges[node][]
	#
	#
	# def dist(self, city1, city2):
	# 	if self.edges[city1._index][city2._index] == True:
	# 		y = city1._y - city2._y
	# 		x = city1._x - city2._x
	#
	# 		dist = np.sqrt(y ** 2 + x ** 2)
	#
	# 		return dist
	# 	else:
	# 		return math.inf