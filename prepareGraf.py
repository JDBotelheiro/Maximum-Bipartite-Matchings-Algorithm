import networkx as nx
import random
from networkx import Graph
from networkx.algorithms import bipartite

def prepareGraf(n1,n2,p):

	SET=[]
	G=bipartite.random_graph(n1,n2,p)

	for edge in G.edges:
		tuple=(edge[0],edge[1])
		SET.append(tuple)

	return SET

prepareGraf(100,100,0.7)
