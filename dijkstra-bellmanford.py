from collections import defaultdict
from itertools import product, chain
import json
import networkx as nx
import pylab
import time
class DijkstraNegativeWeightException(Exception):
    pass

class DijkstraDisconnectedGraphException(Exception):
    pass

class Graph:
    def __init__(self, graph_data, source):
        self._graph = defaultdict(dict, graph_data)
        self._check_edge_weights()
        self.reset_source(source)
        self._solved = False

    @property
    def edges(self):
        return [(i, j) for i in self._graph for j in self._graph[i]]

    @property
    def nodes(self):
        return list(set(chain(*self.edges)))

    def _check_source_in_nodes(self):
        msg = 'Source node \'{}\' not in graph.'
        if self._source not in self.nodes:
            raise ValueError(msg.format(self._source))

    def _check_edge_weights(self):
        msg = 'Graph has negative weights, but weights must be non-negative.'
        if any(self._graph[i][j] < 0 for (i, j) in self.edges):
            raise DijkstraNegativeWeightException(msg)

    def reset_source(self, source):
        self._source = source
        self._check_source_in_nodes()
        self._solution_x = []
        self._solution_x_Bellman_Ford = []
        self._solution_z = {source: 0}
        self._solution_z_Bellman_Ford = {source: 0}
        self._visited = set([source])
        self._unvisited = set()
        for key, val in self._graph.items():
            self._unvisited.add(key)
            self._unvisited.update(val.keys())
        self._unvisited.difference_update(self._visited)
        self._solved = False

    def Dijkstra_run(self):

        weight_candidates = self._graph[self._source].copy()
        node_candidates = dict(product(weight_candidates.keys(),
                                       (self._source,)))
        while node_candidates:
            j = min(weight_candidates, key=weight_candidates.get)
            weight_best, i = weight_candidates.pop(j), node_candidates.pop(j)
            for k in self._graph[j].keys() & self._unvisited:
                weight_next = self._graph[j][k]
                if (k not in node_candidates
                        or weight_candidates[k] > weight_best + weight_next):
                    weight_candidates[k] = weight_best + weight_next
                    node_candidates[k] = j
            self._solution_x.append((i, j))
            self._solution_z[j] = weight_best
            self._visited |= {j}
            self._unvisited -= {j}
        self._solved = True



    def Dijkstra_path_to(self, target):
        if self._source in self._visited and target in self._unvisited:
            msg = 'No path from {} to {}; graph is disconnected.'
            msg = msg.format(self._visited, self._unvisited)
            raise DijkstraDisconnectedGraphException(msg)    
        solution = self._solution_x.copy()
        path = []
        while solution:
            i, j = solution.pop()
            if j == target:
                path.append((i, j))
                break
        while solution:
            i_prev, _, i, j = *path[-1], *solution.pop()
            if j == i_prev:
                path.append((i, j))
                if i == self._source:
                    break
        return list(reversed(path)), self._solution_z[target]

    def Dijkstra_visualize(self, source=None, target=None):
        if (source is not None and source != self._source):
            self.reset_source(source)
        if not self._solved:
            msg = 'No path from {} to {}; graph is disconnected.'
            msg = msg.format(self._source, target)
            raise DijkstraDisconnectedGraphException(msg) 
        if target is not None:
            path, _ = self.Dijkstra_path_to(target=target)
        else:
            path = self._solution_x
        edgelist = self.edges
        nodelist = self.nodes
        nxgraph = nx.DiGraph()
        nxgraph.add_edges_from(edgelist)
        weights = {(i, j): self._graph[i][j] for (i, j) in edgelist}
        found = list(chain(*path))
        ncolors = ['springgreen' if node in found else 'lightcoral'
                   for node in nodelist]
        ecolors = ['dodgerblue' if edge in path else 'black'
                   for edge in edgelist]
        sizes = [1 if edge in path else 1 for edge in edgelist]
        pos = nx.spring_layout(nxgraph)
        nx.draw_networkx(nxgraph, pos=pos,
                         nodelist=nodelist, node_color=ncolors,
                         edgelist=edgelist, edge_color=ecolors, width=sizes,font_size=6)
        nx.draw_networkx_edge_labels(nxgraph, pos=pos, edge_labels=weights, font_size=6)
        pylab.figure(1,figsize=(720,720))
        pylab.title('Graph with Dijkstra Shortest Paths')
        pylab.show()

    def Bellman_Ford_run(self):

        # Step 1: Prepare the distance and predecessor for each node
        graph = self._graph
        source = self._source
        distance, predecessor = dict(), dict()
        for node in graph:
            distance[node], predecessor[node] = float('inf'), None
        distance[source] = 0

        # Step 2: Relax the edges
        for _ in range(len(graph) - 1):
            for node in graph:
                for neighbour in graph[node]:
                    # If the distance between the node and the neighbour is lower than the current, store it
                    if distance[neighbour] > distance[node] + graph[node][neighbour]:
                        distance[neighbour], predecessor[neighbour] = distance[node] + graph[node][neighbour], node

        # Step 3: Check for negative weight cycles
        for node in graph:
            for neighbour in graph[node]:
                assert distance[neighbour] <= distance[node] + graph[node][neighbour], "Negative weight cycle."
        self._solution_x_Bellman_Ford = [(v,k) for k, v in predecessor.items()]
        self._solution_z_Bellman_Ford = distance
        self._solved = True


    def Bellman_Ford_path_to(self, target):
        solution = self._solution_x_Bellman_Ford.copy()
        path = []
        while solution:
            i, j = solution.pop()
            if j == target:
                path.append((i, j))
                break
        while solution:
            i_prev, _, i, j = *path[-1], *solution.pop()
            if j == i_prev:
                path.append((i, j))
                if i == self._source:
                    break
        return list(reversed(path)), self._solution_z[target]
    
    def Bellman_Ford_visualize(self, source=None, target=None):
        if (source is not None and source != self._source):
            self.reset_source(source)
        if not self._solved:
            msg = 'No path from {} to {}; graph is disconnected.'
            msg = msg.format(self._source, target)
            raise DijkstraDisconnectedGraphException(msg)
        if target is not None:
            path, _ = self.Bellman_Ford_path_to(target=target)
        else:
            path = self._solution_x_Bellman_Ford
        edgelist = self.edges
        nodelist = self.nodes
        nxgraph = nx.DiGraph()
        nxgraph.add_edges_from(edgelist)
        weights = {(i, j): self._graph[i][j] for (i, j) in edgelist}
        found = list(chain(*path))
        ncolors = ['springgreen' if node in found else 'lightcoral'
                   for node in nodelist]
        ecolors = ['dodgerblue' if edge in path else 'black'
                   for edge in edgelist]
        sizes = [1 if edge in path else 1 for edge in edgelist]
        pos = nx.spring_layout(nxgraph)
        nx.draw_networkx(nxgraph, pos=pos,
                         nodelist=nodelist, node_color=ncolors,
                         edgelist=edgelist, edge_color=ecolors, width=sizes,font_size=6)
        nx.draw_networkx_edge_labels(nxgraph, pos=pos, edge_labels=weights, font_size=6)
        pylab.figure(1,figsize=(720,720))
        pylab.title('Graph with Bellman-Ford Shortest Paths')
        pylab.show()

#Run with data
with open('data\graph.json') as json_file:
    graph_data = json.load(json_file)   

#Dijkstra
algo = Graph(graph_data=graph_data, source='BachKhoa')
algo.Dijkstra_run()
print("Dijkstra_run:" + " " +str(algo.Dijkstra_path_to("ChoBenThanh")))
algo.Dijkstra_visualize(source='BachKhoa', target='ChoBenThanh')

#Bellman-Ford
algo.Bellman_Ford_run()
print("Bellman_Ford_run:" + " " +str(algo.Dijkstra_path_to("ChoBenThanh")))
algo.Bellman_Ford_visualize(source='BachKhoa', target='ChoBenThanh')

