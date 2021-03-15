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
	startingCity = None
	finalCity = None
	def greedy(self, time_allowance=60.0):
		start_time = time.time()
		# initialize unvisited[] to all cities
		unvisited = self._scenario.getCities().copy()  # would it be better to just have an array of
		# sizeof(unvisited) and initialize all values to 0?
		previousPointers = []
		for c in unvisited:
			previousPointers.append(self.cityData(None, c))

		startingCity = unvisited[0]

		route = []
		currentCity = startingCity
		while len(unvisited) != 0:
			unvisited.remove(currentCity)
			route.append(currentCity)
			# find shortest path to city2 from current city1
			nextCity = self.findClosestCity(currentCity, unvisited)
			if nextCity == currentCity: finalCity = currentCity

			# update previous pointers
			previousPointers[nextCity._index].previous = currentCity

			currentCity = nextCity

		results = {}
		foundTour = False
		bssf = TSPSolution(route)
		count = len(route)
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

	def findClosestCity(self, currentCity, unvisited):

		# loop through all edges of city1
		edges = self._scenario.getEdges()[currentCity._index]
		cities = self._scenario.getCities()
		shortestDistance = 999999
		closestCity = currentCity
		for i in range(len(edges)):
			if edges[i] and (cities[i] in unvisited ):
				distance = self.calculateDistance(cities[i], currentCity)
				if distance < shortestDistance:
					shortestDistance = distance
					closestCity = cities[i]

		return closestCity



		# if destination city is in unvisited
		# update closest city

		# return closest city
		pass


	def calculateDistance(self, city1, city2):
		y = city1._y - city2._y
		x = city1._x - city2._x

		dist = np.sqrt(y**2 + x**2)

		return dist

	
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
		pass
		



