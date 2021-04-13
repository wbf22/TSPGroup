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

class State:
	def __init__( self, lev, cancelRow, cancelCol, lb, mat ):
		self.level = lev
		self.cancelR = cancelRow
		self.cancelC = cancelCol
		self.lowerbound = lb
		self.matrix = mat
		self.weight = lb - lev*456
		self.next = -1


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

	reducedMatrix = []
	queue = []

	def branchAndBound(self, time_allowance=60.0):
		self.results = self.greedy()
		self.cities = self._scenario.getCities()
		self.ncities = len(self.cities)
		self.bssf = self.results['cost']
		self.intermediateSolutions = 0
		self.total = 0
		self.pruned = 0
		self.max = 0
		self.count = 0
		self.lastState = None
		foundTour = False

		# Make the matrix
		self.reducedMatrix = [[i.costTo(j) for j in self.cities] for i in self.cities]

		# Reduce the matrix
		lowBound = self.reduceRows()
		lowBound += self.reduceCols()

		# insert the first state into the queue
		self.insertMat(State(0, [], [], lowBound, self.reducedMatrix.copy()))
		self.queue[0].next = 0

		start_time = time.time()

		# here we loop through checking each state in the queue.
		while len(self.queue) > 0 and time.time() - start_time < time_allowance:
			cur = self.queue.pop(0)
			self.searchState(cur)

		end_time = time.time()

		# prep the results and return
		# if searchState() didn't find a result, just return the original greedy result
		if self.lastState != None:
			route = []
			for i in self.lastState.cancelR:
				route.append(self.cities[i])
			soln = TSPSolution(route)

			self.results['cost'] = self.bssf
			self.results['time'] = end_time - start_time
			self.results['count'] = self.count
			self.results['soln'] = soln
		else:
			self.results['count'] = 0
		self.results['max'] = self.max
		self.results['total'] = self.total
		self.results['pruned'] = self.pruned
		return self.results

	def searchState(self, cur):
		# skip this state if the lowerbound is greater than the current bssf
		if cur.lowerbound > self.bssf:
			self.pruned += 1
			return

		row = cur.next
		for col in range(self.ncities):
			self.total += 1

			# skip this cell if the column is alread done or if the cell value is np.inf
			if cur.matrix[row][col] == np.inf:
				self.pruned += 1
				continue
			if col in cur.cancelC:
				continue
			mat = [cur.matrix[i].copy() for i in range(self.ncities)]
			cost = mat[row][col]
			# cancel out row and col
			for x in range(self.ncities):
				mat[row][x] = np.inf
				mat[x][col] = np.inf
			mat[col][row] = np.inf
			canR = cur.cancelR.copy()
			canR.append(row)
			canC = cur.cancelC.copy()
			canC.append(col)

			# reduce matrix
			cost += self.reduceRows(mat, canR)
			cost += self.reduceCols(mat, canC)

			# check if complete state
			if (cur.level + 1 == self.ncities):
				self.intermediateSolutions += 1
				# if it is, update the bssf
				if cost + cur.lowerbound < self.bssf:
					self.count += 1
					self.bssf = cost + cur.lowerbound
					self.lastState = State(cur.level + 1, canR, canC, cur.lowerbound + cost, mat)
				else:
					self.pruned += 1
				continue

			# prune this state if it is greater than the bssf
			if cost > self.bssf:
				self.pruned += 1
				continue

			# add the current state to the queue
			boomer = State(cur.level + 1, canR, canC, cur.lowerbound + cost, mat)
			boomer.next = col
			self.insertMat(boomer)

	# Function to insert state into queue
	def insertMat(self, state):
		# if the queue is empty, just insert
		if len(self.queue) == 0:
			self.queue.append(state)
			return

		inserted = False
		# loop through until the current state is in the right spot, and insert
		for k in range(len(self.queue)):
			if state.weight <= self.queue[k].weight:
				self.queue.insert(k, state)
				inserted = True
				break

		# if the current state is greater than all items in the queue, insert at end
		if not inserted:
			self.queue.append(state)

		# keep track of max length of queue
		if len(self.queue) > self.max:
			self.max = len(self.queue)
		return

	# this is a function to reduce the rows of a matrix
	def reduceRows(self, matrix=None, cancelled=[]):
		# if were not given a matrix, assume we are reducing the original matrix
		if matrix == None:
			matrix = self.reducedMatrix
		boom = 0;
		for i in range(self.ncities):
			if i in cancelled:
				continue
			temp = self.getMinRow(matrix[i])
			boom += temp
			if temp == 0 or temp == np.inf:
				continue
			else:
				for j in range(self.ncities):
					matrix[i][j] -= temp
		return boom

	# this is a function to reduce the columns of a matrix
	def reduceCols(self, matrix=None, cancelled=[]):
		# if were not given a matrix, assume we are reducing the original matrix
		if matrix == None:
			matrix = self.reducedMatrix
		boom = 0;
		for i in range(self.ncities):
			if i in cancelled:
				continue
			temp = self.getMinCol(i)
			boom += temp
			if temp == 0 or temp == np.inf:
				continue
			else:
				for j in range(self.ncities):
					matrix[j][i] -= temp
		return boom

	# this function just gets the smallest value in the row
	def getMinRow(self, listC):
		msf = np.inf
		for i in listC:
			if (i < msf):
				msf = i
		return msf

	# this function just gets the smallest value in the col
	def getMinCol(self, i, matrix=None):
		if matrix == None:
			matrix = self.reducedMatrix
		msf = np.inf
		for j in range(self.ncities):
			boomer = matrix[j][i]
			if (boomer < msf):
				msf = boomer
		return msf



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
		bestSolutionSoFar = None

		#run 100 times and keep the best solution
		for i in range(100):
			# pick arbitrary node
			# find node outside C closest to a node in C called K
			# find edge in C that dik+dkj-dij is minimal
			# Make new cycle replacing ij with ik and kj
			# keeping going until C has all vertices
			self.nodes = self._scenario.getCities().copy()
			self.edges = self._scenario.getEdges().copy()
			staringNode = random.choice(self.nodes)
			self.path = [staringNode]
			self.nodes.remove(staringNode)

			start = time.time()
			while len(self.nodes) > 0:
				self.addNode()
				if time.time() - start > 0.5: break; #will time out if no new nodes can be added




			bssf = TSPSolution(self.path)
			count = len(self.path)
			if count == len(self._scenario.getCities()):
				#check to see if it's the best result so far
				if bestSolutionSoFar == None: bestSolutionSoFar = bssf
				if bssf.cost < bestSolutionSoFar.cost: bestSolutionSoFar = bssf
				#print(bssf.cost)

		######
		end_time = time.time()

		foundTour = False
		if bestSolutionSoFar.cost < np.inf:
			# Found a valid route
			foundTour = True
		results = {}
		results['cost'] = bestSolutionSoFar.cost if foundTour else 999999
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bestSolutionSoFar
		results['max'] = None
		results['total'] = None
		results['pruned'] = None



		return results

	def addNode(self):
		#using the "5: Random Insertion" algorithm from the website https://stemlounge.com/animated-algorithms-for-the-traveling-salesman-problem/
		index = -1
		#keep going until you find a valid node to add
		start = time.time()
		while index == -1:
			newNode = self.nextNodeToInsert()
			index = self.whereToInsertNode(newNode)
			if time.time() - start > 0.5: break; #will time out if node cannot be inserted

		self.path.insert(index, newNode)
		self.nodes.remove(newNode)

	#finds two points in between which a node can be inserted
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

	#finds a node that is closest to a random node already in the path
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