#https://en.wikipedia.org/wiki/Matching_(graph_theory)
#https://en.wikipedia.org/wiki/Maximum_cardinality_matching
import math
import random
import sys
import networkx as nx
import random
from networkx import Graph
from networkx.algorithms import bipartite



def get_path_list(path, source, target):

	result = [path[target]]
	curr_node = path[target].source

	while curr_node != source:
		result.append(path[curr_node])
		curr_node = path[curr_node].source

	result.reverse()

	return result


def get_link_tuples_from_input():

	#num_nodes = int(input("Number of nodes: "))
	#num_links = int(input("Number of links: "))

	num_nodes = int(input())
	num_links = int(input())
	
	link_tuples = []

	for i in range(num_links):
		
		line = input().split()
		link_tuple = (int(line[0]), int(line[1]))
		
		#weighted link
		if(len(line) == 3):
			link_tuple += (int(line[2]),)
		
		link_tuples.append(link_tuple)

	return num_nodes, link_tuples


def add_super_source_sink(link_tuples, num_nodes, super_link_weight):

	super_source = num_nodes
	super_sink = num_nodes + 1

	for node in range(num_nodes):

		#if node is on the "left side" of the matroid
		if node%2 == 0:
			#add link from the super source to the node
			link_tuples.append( (super_source, node, super_link_weight) )


		#if node is on the "right side" of the matroid
		else:
			#add link from the node to the super sink
			link_tuples.append( (node, super_sink, super_link_weight) )

	return num_nodes + 2, super_source, super_sink


#directed link
class Link:

	def __init__(self, source, target, weight = 1):
		self.source = source
		self.target = target
		self.weight = weight #capacity
		self.flow = 0
		self.inverse = None

	#used for sorting
	def __lt__(self, other):
		#sort by source then by target
		if self.source == other.source:
			return self.target < other.target

		else:
			return self.source < other.source

	#to string
	def __repr__(self):
		return f"<link {self.source}->{self.target} : {self.flow}/{self.weight}>"

	def has_capacity(self):
		return self.weight - self.flow > 0


class Graph:

	#link_tuples is a list of tuples -> (source, target, weight (optional))
	def __init__(self, num_nodes, link_tuples):
		self.num_nodes = num_nodes
		self.links = []
		self.links_start_index = [0] * num_nodes

		for link in link_tuples:
			#unweighted link
			if len(link) == 2:
				new_link = Link(link[0], link[1])
				inverse = Link(link[1], link[0], 0)


			#weighted link
			else:
				new_link = Link(link[0], link[1], link[2])
				inverse = Link(link[1], link[0], 0)

			new_link.inverse = inverse
			inverse.inverse = new_link

			self.links.append(new_link)
			self.links.append(inverse)

		self.compute_links_start_index()


	def compute_links_start_index(self):
		#this is O(n log(n)), maybe change to count sort for O(n)
		self.links.sort()

		curr_source = 0

		i = 0
		while i < len(self.links):
			if(self.links[i].source != curr_source):
				curr_source += 1
				self.links_start_index[curr_source] = i

			i += 1

		i -= 1
		while curr_source < self.num_nodes:
			self.links_start_index[curr_source] = i
			curr_source += 1



	#only returns links with capacity > 0
	def get_links_by_source(self, source):

		index = self.links_start_index[source]
		links = []

		while(index < len(self.links) and self.links[index].source == source):
			if(self.links[index].has_capacity()):
				links.append(self.links[index])

			index += 1

		return links


	def BFS(self, source, target):

		visited = [False] * self.num_nodes
		queue = [source]
		path = [None] * self.num_nodes
		found_path = False

		while len(queue) != 0:
			#pop the first node of the queue
			node = queue.pop(0)

			#reached target
			if node == target:
				found_path = True
				break

			if not visited[node]:
				#mark this note as visited
				visited[node] = True
				links = self.get_links_by_source(node)

				for link in links:
					neighbour = link.target
					if path[neighbour] == None:
						path[neighbour] = link
						queue.append(neighbour)

		#no path to target was found, return empty list
		if not found_path:
			return []


		return get_path_list(path, source, target)


	def Bellman_Ford(self, source, target):

		path = [None] * self.num_nodes
		distance = [math.inf] * self.num_nodes
		distance[source] = 0

		#relax all links V-1 times
		i = 0
		while( i < self.num_nodes - 1):
			for link in self.links:
				if (link.has_capacity() and distance[link.source] + link.weight < distance[link.target]):
					distance[link.target] = distance[link.source] + link.weight
					path[link.target] = link

			i += 1

		#no path to target was found, return empty list
		if path[target] == None:
			return []


		#check for negative-weight cycles
		for link in self.links:
			if link.has_capacity() and distance[link.source] + link.weight < distance[link.target]:
				#negative weight cycle detected
				return []


		return get_path_list(path, source, target)


	def FordFulkerson(self, source, sink, path_alg):

		path = path_alg(source, sink)
		max_flow = 0

		while len(path) != 0:

			#find the mininum residual capacity of the links along the path
			flow = min(link.weight - link.flow for link in path)

			#add path flow to max_flow
			max_flow += flow
			
			for link in path:
				link.flow += flow
				link.inverse.flow -= flow
			
			path = path_alg(source, sink)
			
		return max_flow


	def get_matching(self, super_source, super_sink):

		for link in self.links:
			if 	(link.source != super_source and link.source != super_sink and link.target != super_source and link.target != super_sink and
				link.weight != 0 and link.flow > 0):
				
				print(link)


	def reverse_weights(self, super_source, super_sink):

		max_weight = max(link.weight for link in self.links) + 1

		for link in self.links:
			if 	(link.source != super_source and link.source != super_sink and link.target != super_source and link.target != super_sink and
			link.weight != 0):
				link.weight = -(link.weight - max_weight)


def gen_random_bipartite(num_nodes, num_links, weighted_links = False, max_weight = 10000):

	left_nodes = range(0, num_nodes, 2)
	right_nodes = range(1, num_nodes, 2)

	link_tuples = []

	for i in range(num_links):

		if weighted_links:
			link_tuples.append( (random.choice(left_nodes), random.choice(right_nodes), random.randint(1, max_weight)) )
		
		else:
			link_tuples.append( (random.choice(left_nodes), random.choice(right_nodes)) )
	
	return link_tuples


def compute_maximum_size_bipartite_matching(link_tuples = None, num_nodes = 0):

	if(link_tuples == None):
		num_nodes, link_tuples = get_link_tuples_from_input()

	num_nodes, super_source, super_sink = add_super_source_sink(link_tuples, num_nodes, 1)
	g = Graph(num_nodes, link_tuples)

	g.FordFulkerson(super_source, super_sink, g.BFS)
	g.get_matching(super_source, super_sink)


def compute_maximum_weight_bipartite_matching(link_tuples = None, num_nodes = 0):

	if(link_tuples == None):
		num_nodes, link_tuples = get_link_tuples_from_input()

	num_nodes, super_source, super_sink = add_super_source_sink(link_tuples, num_nodes, 1)
	g = Graph(num_nodes, link_tuples)
	g.reverse_weights(super_source, super_sink)

	g.FordFulkerson(super_source, super_sink, g.Bellman_Ford)
	g.get_matching(super_source, super_sink)


#compute_maximum_size_bipartite_matching()
#compute_maximum_weight_bipartite_matching()

option = int(input())

if option == 1:
	compute_maximum_size_bipartite_matching()

elif option == 2:
	compute_maximum_weight_bipartite_matching()

else:
	print(f"Option {option} does not exist")



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # LINK AND GRAPH STRUCTURE # # # # # # # # # # # # # # # # # # # # # # 
class Link:

	def __init__(self, source, target, weight = 1):
		self.source = source
		self.target = target


class Graph:

	#link_tuples is a list of tuples -> (source, target, weight (optional))
	def __init__(self, num_nodes, link_tuples):
		self.num_nodes = num_nodes
		self.links = []

		for link in link_tuples:
			new_link = Link(link[0], link[1])

			self.links.append(new_link)


	def BFS(self,X1,X2):
		path={}
		Q=[]
		for x in X1:
			Q.append(x)
		while len(Q)!=0:
			x=Q.pop(0)
			if x in X2:
				P=getPath(path,X1,X2)
				return P
			for link in self.links:
				if (link.source==x and link.target not in path.keys()):
					Q.append(link.target)
					path[link.target]=x
		return P

def getPath(path,X1,X2):
	P=[]
	shortest_path=[]
	shortest=len(X1)+len(X2)
	for x in X2:
		curr=x
		while(curr not in X1 and curr in path):
			P.append((curr))
			curr=path[curr]
		if curr in X1:
			P.append((curr))
			if len(P)<=shortest:
				shortest=len(P)
				shortest_path=P.copy()
	return shortest_path


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # TO CREATE THE MATROIDS # # # # # # # # # # # # # # # # # # # # # # 

def f1(I,x,y):	
	source=y[0]
	for i in I:
		if x!=i and i[0]==source:
				return True
	return False

def f2(I,x,y):
	target=y[1]
	for i in I:
		if i!=x and i[1]==target:
				return True
	return False

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # TO CREATE THE EXCHANGE GRAPH # # # # # # # # # # # # # # # # # # # # # 
 
def DM(E,I,nr_nodes,f1,f2):
	Links=[]
	i=0
	for x in I:
		for y in E:
			if (not f1(I,x,y)):
				Links.append((x,y))
			if (not f2(I,x,y)):
				Links.append((y,x))
	D=Graph(nr_nodes,Links)
	return D


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # MATROID INTERSECTION # # # # # # # # # # # # # # # # # # # # # # # # # 

def augment(E,I,P):
	case=0
	for link in P:
		if case==0:
			I[link]=None
			E.remove(link)
			case=1
		elif case==1:
			del(I[link])
			E.append(link)
			case=0		
	return I


def MatroidIntersection(groundSet, f1, f2):

	I={}
	nr_nodes=len(groundSet)**2
	E=groundSet.copy()
	x=E.pop(0)
	I[x]=None

	while(True):
		D=DM(E,I,nr_nodes,f1,f2)
		X1=[]
		X2=[]
		for x in E:
			if (not f1(I,x,x)):
				X1.append(x)
			if (not f2(I,x,x)):
				X2.append(x)
		if X1==[] or X2==[]:
			break
		P=D.BFS(X1,X2)
		if len(P)==0:
			break
		else:
			augment(E,I,P)
	return(I)

Gset=[(0,5),(0,8),(1,6),(1,7),(2,5),(2,9),(3,5),(3,8),(4,8),(4,7),(4,9)]
G=bipartite.random_graph(10,10,0.5)
SET=[]
for edge in G.edges:
	tuple=(edge[0],edge[1])
	SET.append(tuple)

I=MatroidIntersection(SET,f1,f2)
print(I)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # PREPARE GRAPH # # # # # # # # # # # # # # # # # # # # # def prepareGraf(n1,n2,p):

	SET=[]
	G=bipartite.random_graph(n1,n2,p)

	for edge in G.edges:
		tuple=(edge[0],edge[1])
		SET.append(tuple)

	return SET