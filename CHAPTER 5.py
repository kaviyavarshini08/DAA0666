#1
def maxCoins(piles):
    piles.sort(reverse=True)
    max_coins = 0
    n = len(piles) // 3 
    for i in range(n):
        max_coins += piles[2 * i + 1]
    
    return max_coins

piles1 = [2, 4, 1, 2, 7, 8]
print(maxCoins(piles1))  


#2
def minCoins(coins, target):
    coins.sort()
    current_sum = 0
    additions = 0
    
    for coin in coins:
        while current_sum + 1 < coin:
            current_sum += (current_sum + 1)
            additions += 1
            if current_sum >= target:
                return additions
        current_sum += coin
        if current_sum >= target:
            return additions
    
    while current_sum < target:
        current_sum += (current_sum + 1)
        additions += 1
        
    return additions


coins1 = [1, 4, 10]
target1 = 19
print("2) ",minCoins(coins1, target1))  
 

#3
def canAssign(jobs, k, max_time):
    workers = [0] * k
    
    def backtrack(index):
        if index == len(jobs):
            return True
        for i in range(k):
            if workers[i] + jobs[index] <= max_time:
                workers[i] += jobs[index]
                if backtrack(index + 1):
                    return True
                workers[i] -= jobs[index]
            if workers[i] == 0:
                break
        return False
    
    jobs.sort(reverse=True)
    return backtrack(0)

def minimumTimeRequired(jobs, k):
    left, right = max(jobs), sum(jobs)
    
    while left < right:
        mid = (left + right) // 2
        if canAssign(jobs, k, mid):
            right = mid
        else:
            left = mid + 1
    
    return left

jobs1 = [3, 2, 3]
k1 = 3
print("3) ",minimumTimeRequired(jobs1, k1))  
#4
from bisect import bisect_left

def jobScheduling(startTime, endTime, profit):
    jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
    dp = [(0, 0)]  
    for s, e, p in jobs:
        i = bisect_left(dp, (s,))
        if dp[i-1][1] + p > dp[-1][1]:
            dp.append((e, dp[i-1][1] + p))
    
    return dp[-1][1]


startTime1 = [1, 2, 3, 3]
endTime1 = [3, 4, 5, 6]
profit1 = [50, 10, 40, 70]
print("4) ",jobScheduling(startTime1, endTime1, profit1))  

#5
import heapq

def dijkstra(n, graph, source):
    dist = [float('inf')] * n
    dist[source] = 0
    min_heap = [(0, source)]
    
    while min_heap:
        current_dist, u = heapq.heappop(min_heap)
        
        if current_dist > dist[u]:
            continue
        
        for v in range(n):
            if graph[u][v] != float('inf') and current_dist + graph[u][v] < dist[v]:
                dist[v] = current_dist + graph[u][v]
                heapq.heappush(min_heap, (dist[v], v))
    
    return dist


n1 = 5
graph1 = [[0, 10, 3, float('inf'), float('inf')], 
          [float('inf'), 0, 1, 2, float('inf')], 
          [float('inf'), 4, 0, 8, 2],
          [float('inf'), float('inf'), float('inf'), 0, 7], 
          [float('inf'), float('inf'), float('inf'), 9, 0]]
source1 = 0
print("5) ",dijkstra(n1, graph1, source1))  
#6
import heapq
from collections import defaultdict

def dijkstra(n, edges, source, target):
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  
    
    dist = {i: float('inf') for i in range(n)}
    dist[source] = 0
    min_heap = [(0, source)]
    
    while min_heap:
        current_dist, u = heapq.heappop(min_heap)
        
        if u == target:
            return current_dist
        
        if current_dist > dist[u]:
            continue
        
        for v, weight in graph[u]:
            if current_dist + weight < dist[v]:
                dist[v] = current_dist + weight
                heapq.heappush(min_heap, (dist[v], v))
    
    return float('inf')


n1 = 6
edges1 = [(0, 1, 7), (0, 2, 9), (0, 5, 14), (1, 2, 10), (1, 3, 15), 
          (2, 3, 11), (2, 5, 2), (3, 4, 6), (4, 5, 9)]
source1 = 0
target1 = 4
print("6) ",dijkstra(n1, edges1, source1, target1))  


#7
import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_codes(characters, frequencies):
    heap = [HuffmanNode(characters[i], frequencies[i]) for i in range(len(characters))]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    
    root = heap[0]
    huffman_code = {}
    
    def generate_codes(node, current_code):
        if node is None:
            return
        if node.char is not None:
            huffman_code[node.char] = current_code
            return
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")
    
    generate_codes(root, "")
    return sorted(huffman_code.items())


characters1 = ['a', 'b', 'c', 'd']
frequencies1 = [5, 9, 12, 13]
print("7) ",huffman_codes(characters1, frequencies1)) 



#8
def decode_huffman(characters, frequencies, encoded_string):
    heap = [HuffmanNode(characters[i], frequencies[i]) for i in range(len(characters))]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    
    root = heap[0]
    decoded_string = ""
    current_node = root
    
    for bit in encoded_string:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        
        if current_node.char is not None:
            decoded_string += current_node.char
            current_node = root
    
    return decoded_string


characters1 = ['a', 'b', 'c', 'd']
frequencies1 = [5, 9, 12, 13]
encoded_string1 = '1101100111110'
print("8) ",decode_huffman(characters1, frequencies1, encoded_string1)) 




#9
def max_weight_loaded(weights, max_capacity):
    weights.sort(reverse=True)
    current_weight = 0
    
    for weight in weights:
        if current_weight + weight <= max_capacity:
            current_weight += weight
    
    return current_weight


weights1 = [10, 20, 30, 40, 50]
max_capacity1 = 60
print("9) ",max_weight_loaded(weights1, max_capacity1)) 



#10
def min_containers(weights, max_capacity):
    weights.sort(reverse=True)
    containers = 0
    current_capacity = 0
    
    for weight in weights:
        if current_capacity + weight > max_capacity:
            containers += 1
            current_capacity = weight
        else:
            current_capacity += weight
    
    if current_capacity > 0:
        containers += 1
    
    return containers

# Test Case 1
weights1 = [5, 10, 15, 20, 25, 30, 35]
max_capacity1 = 50
print("10) ",min_containers(weights1, max_capacity1))  



#11
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def kruskal(n, edges):
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(n)
    mst = []
    total_weight = 0
    
    for u, v, weight in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append((u, v, weight))
            total_weight += weight
    
    return mst, total_weight


n1 = 4
edges1 = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
mst1, total_weight1 = kruskal(n1, edges1)
print(f"Edges in MST: {mst1}")  # Output: [(2, 3, 4), (0, 3, 5), (0, 1, 10)]
print(f"Total weight of MST: {total_weight1}")  # Output: 19


#12
def is_unique_mst(n, edges, given_mst):
    def find_mst():
        uf = UnionFind(n)
        mst = []
        for u, v, weight in edges:
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                mst.append((u, v, weight))
        return mst
    
    edges.sort(key=lambda x: x[2])
    given_mst_set = set(given_mst)
    
    first_mst = find_mst()
    if set(first_mst) == given_mst_set:
        return True, first_mst, None
    
    for u, v, w in first_mst:
        if (u, v, w) not in given_mst_set:
            alternate_mst = find_mst()
            return False, first_mst, alternate_mst
    
    return True, first_mst, None

n1 = 4
edges1 = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
given_mst1 = [(2, 3, 4), (0, 3, 5), (0, 1, 10)]
unique1, mst1, another_mst1 = is_unique_mst(n1, edges1, given_mst1)
print(f"Is the given MST unique? {unique1}")
