from typing import List, Dict, Tuple
import collections
import sys

class network:
    # G = (V,E) with c for e in E
    # construct network from edge(u,v,c)
    def __init__(self, edges: List[int], s: int, t: int):
        self.s = s
        self.t = t
        self.vertices = set()
        self.g = collections.defaultdict(list) # adj nodes: u to [v,w]
        self.capacities = collections.defaultdict(int)  # (u,v) : cuv
        self.f = collections.defaultdict(int) # (u,v) : fuv
        
        self.const_g_from_e(edges)
        
        
    def const_g_from_e(self, edges: List[List[int]]):
        # edge: u to v == v to u
        for u, v, c in edges:
            self.vertices.add(v)
            self.vertices.add(u)
            self.g[u].append(v)
            self.capacities[(u,v)] = c
#             self.f[(u,v)] = 0
            
    def build_residual_g(self, flow: Dict[Tuple[int,int], int]):
        
        # residual network Gf, share V with G, process E
        self.gf = collections.defaultdict(list)
        self.gf_c = collections.defaultdict(int)
        
        for edge,cap_uv in self.capacities.items():
            u,v = edge
            if self.f[edge] < cap_uv:
                # forward edge
                self.gf_c[edge] = cap_uv - self.f[edge]
                self.gf[u].append(v)
            if self.f[edge] > 0:
                # backward edge
                self.gf_c[(v,u)] = self.f[edge]
                self.gf[v].append(u)
    
    def find_path(self, graph: Dict[int, List[int]], 
                  capacities: Dict[Tuple[int,int], int], 
                  s: int, t: int):
        
        q = [s]
        visited = set()
        visited.add(s)
        prev = {}
        
        while q:
            next_q = []
            for node in q:
#                 print("process node: ", node)
                for nei in graph[node]:
                    if nei in visited:
                        continue
                    if capacities[(node,nei)] == 0:
                        continue
                    if nei == t:
                        prev[nei] = node
                        return prev
                    visited.add(nei)
                    prev[nei] = node
                    next_q.append(nei)
            q = next_q
        return None # not found
    
    def build_flow(self, prev: Dict[int,int], 
                   capacities: Dict[Tuple[int,int], int],
                   s: int, t: int):
        # output: flow[(u,v)] list of edges
        #         flow min-capacity: min of cap
        flow = []
        min_cap = sys.maxsize
        node = t
        while node != s:
            flow.append((prev[node],node))
            min_cap = min(min_cap, capacities[(prev[node],node)])
            node = prev[node]
        return (flow, min_cap)
    
    def augument_flow(self, path: List[Tuple[int,int]], units: int):
        
        for u, v in path:
            if self.capacities[(u,v)] > 0:
                # forward edge, + units
                self.f[(u,v)] += units
            if self.capacities[(v,u)] > 0:
                # backward edge, - units
                self.f[(v,u)] -= units
    
    def max_flow(self):
        while True:
            # step 1: build residual network
            self.build_residual_g(self.f)
            # step 2: find path s-t in Gf
            prev = self.find_path(self.gf, self.gf_c, self.s, self.t)
            if not prev:
                # no path found
                break
            path, cp = self.build_flow(prev, self.gf_c, self.s, self.t)

            # step 3: augument self.f
            self.augument_flow(path, cp)
#             print("current flow: ")
#             print(self.f)
        return {key: val for key, val in self.f.items() if val > 0}
    
    def __repr__(self):
        return "s: " + str(self.s) + " t: " + str(self.t) + \
        "\ng: " + str(self.g) + "\nc: " + str(self.capacities)

if __name__ == "__main__":
    n3 = network([[0,1,7],[0,2,6],[1,3,4],[1,4,2],[2,3,2],[2,4,3],[3,5,9],[4,5,5]],0,5)
    f = n3.max_flow()
    print(f)
