
from operator import itemgetter
from prioritydictionary import priorityDictionary

## @package YenKSP
# Computes K-Shortest Paths using Yen's Algorithm.
#
# Yen's algorithm computes single-source K-shortest loopless paths for a graph 
# with non-negative edge cost. The algorithm was published by Jin Y. Yen in 1971
# and implores any shortest path algorithm to find the best path, then proceeds 
# to find K-1 deviations of the best path.

## Computes K paths from a source to a sink in the supplied graph.
#
# @param graph A digraph of class Graph.
# @param start The source node of the graph.
# @param sink The sink node of the graph.
# @param K The amount of paths being computed.
#
# @retval [] Array of paths, where [0] is the shortest, [1] is the next 
# shortest, and so on.
#
def ksp_yen(graph, node_start, node_end, max_k=9):
    
    distances, previous = dijkstra(graph, node_start)

    A = [{'cost': distances[node_end], 
          'path': path(previous, node_start, node_end)}]
    B = []
    
    if not A[0]['path']: return A
    
    for k in range(1, max_k):
        for i in range(0, len(A[-1]['path']) - 1):
            node_spur = A[k-1]['path'][i]
            path_root = A[k-1]['path'][:i+1]
            edges_removed = []
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i+1]:
                    # cost = graph[curr_path[i]][curr_path[i+1]]["weight"]
                    try:
                        graph.remove_edge(curr_path[i], curr_path[i+1])
                    except:
                        pass
                    try:
                        edges_removed.append([curr_path[i], curr_path[i+1], graph[curr_path[i]][curr_path[i+1]]["weight"]])
                    except:
                        pass
            path_spur = dijkstra(graph, node_spur, node_end)
            
            if path_spur['path']:
                path_total = path_root[:-1] + path_spur['path']
                dist_total = distances[node_spur] + path_spur['cost']
                potential_k = {'cost': dist_total, 'path': path_total}
            
                if not (potential_k in B):
                    B.append(potential_k)
            
            for edge in edges_removed:
                graph.add_edge(edge[0], edge[1], weight=edge[2])
        
        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            A.append(B[0])
            B.pop(0)
        else:
            break
    
    return A

## Computes the shortest path from a source to a sink in the supplied graph.
#
# @param graph A digraph of class Graph.
# @param node_start The source node of the graph.
# @param node_end The sink node of the graph.
#
# @retval {} Dictionary of path and cost or if the node_end is not specified,
# the distances and previous lists are returned.
#
def dijkstra(graph, node_start, node_end=None):
    distances = {}      
    previous = {}       
    Q = priorityDictionary()
    
    for v in graph:
        distances[v] = float('inf')
        previous[v] = None
        Q[v] = float('inf')
    
    distances[node_start] = 0
    Q[node_start] = 0
    
    for v in Q:
        if v == node_end: 
            break
        for u in graph[v]:
            try:
                cost_vu = distances[v] + graph[v][u]["weight"]        
            except:
                pass
            if cost_vu < distances[u]:
                distances[u] = cost_vu
                Q[u] = cost_vu
                previous[u] = v

    if node_end:
        return {'cost': distances[node_end], 
                'path': path(previous, node_start, node_end)}
    else:
        return (distances, previous)

## Finds a paths from a source to a sink using a supplied previous node list.
#
# @param previous A list of node predecessors.
# @param node_start The source node of the graph.
# @param node_end The sink node of the graph.
#
# @retval [] Array of nodes if a path is found, an empty list if no path is 
# found from the source to sink.
#
def path(previous, node_start, node_end):
    route = []

    node_curr = node_end    
    while True:
        route.append(node_curr)
        if previous[node_curr] == node_start:
            route.append(node_start)
            break
        elif previous[node_curr] == None:
            return []
        
        node_curr = previous[node_curr]   
    route.reverse()
    return route

import json
import networkx as nx
import pylab
with open('data\graph.json') as json_file:
    graph_data = json.load(json_file)  
for k, d in graph_data.items():
    for ik in d:
        d[ik] = {'weight': d[ik]}
G = nx.DiGraph(graph_data)
# pos=nx.kamada_kawai_layout(G)
# nx.draw_networkx(G,pos,font_size=6,width=1)
# labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_size=6)
# pylab.figure(1,figsize=(720,720))
# pylab.show()
print(ksp_yen(graph=G,node_start='BachKhoa',node_end="ChoBenThanh",max_k=3))
