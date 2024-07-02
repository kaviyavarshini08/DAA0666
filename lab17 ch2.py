#1 sort
def sort(n):
    if n==[]:
        return None
    for i in range (len(n)):
        for j in range(i,len(n)):
            if( n[i]>n[j]):
                n[i],n[j]=n[j],n[i]
    return n[-1]
n=list(map(int,input().split()))
print(sort(n))

#2 selection sort
def ssort(arr):
    n = len(arr)
    for i in range(n):
        min= i
        for j in range(i+1,n):
            if arr[j]< arr[min]:
                min= j
        arr[i],arr[min]= arr[min],arr[i]
    return arr
arr = list(map(int,input().split()))
print(ssort(arr))

#3 bubble sort
def bsort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0,n-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr
arr = list(map(int,input().split()))
print(bsort(arr))

#4 insertion sort
def isort(arr):
  for i in range(0,len(arr)):
    key = arr[i]
    j = i - 1
    while j >= 0 and arr[j] > key:
      arr[j + 1] = arr[j]
      j = j - 1
    arr[j + 1] = key
  return arr
arr = list(map(int,input().split()))
print(isort(arr))

#5 missing
arr = list(map(int,input().split()))
k = 5
m=max(arr)+1
s=[]
for i in range(1,m):
    if i not in arr:
        s.append(i)
if k<len(arr):
    print(s[k])
    
#6 index of peak elem
def find_peak_element(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid  
        else:
            left = mid + 1
    return left
nums = [1, 2, 3, 1]
print(find_peak_element(nums))   

#7 index of needle in haystack
def str(hay, need):
    if not need:
        return 0
    h= len(hay)
    n= len(need)
    for i in range(h - n + 1):
        if hay[i:i+n] == need:
            return i
    return -1
hay = "sadbutsad"
need = "sad"
print(str(hay, need))

#8 substr in str
def substringWords(words):
    result = set()
    for i in range(len(words)):
        cword = words[i]
        for j in range(len(words)):
            if i != j and cword in words[j]:
                result.add(cword)
                break    
    return list(result)
words = ["mass","as","hero","super hero"]
print(substringWords(words))  

#9 2d closest pair
def brute(points):
    mini= float('inf')
    closest= None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]
            dis = distance(p1, p2)            
            if dis < mini:
                mini = dis
                closest = (p1, p2)                
    return closest, mini
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
point = [(0, 0), (1, 1), (4, 5), (3, 1)]
res , value = brute(point)
print(res)
print(value)

#10 closest
def brute(points):
    mini= float('inf')
    closest= None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]
            dis = distance(p1, p2)            
            if dis < mini:
                mini = dis
                closest = (p1, p2)                
    return closest, mini
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
point = [(0, 0), (1, 1), (4, 5), (3, 1)]
res , value = brute(point)
print(res)
print(value)

#11
#12 tsp
import math
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
import itertools
def tsp(cities):
    n = len(cities)
    if n <= 1:
        return 0, []
    allperms = itertools.permutations(range(1, n))    
    mindis = float('inf')
    shortpath = None
    for perm in allperms:
        path = [0] + list(perm) + [0]
        totaldist = 0
        for i in range(n):
            totaldist += distance(cities[path[i]], cities[path[i + 1]])
        if totaldist < mindis:
            mindis = totaldist
            shortestpath = path
    spc= [cities[idx] for idx in shortestpath]    
    return mindis, spc
cities = [(1, 2), (4, 5), (7, 1), (3, 6)]
mindis, sp = tsp(cities)
print(f"Shortest Distance: {mindis}")
print(f"Shortest Path: {sp}")

#13 cost matrix
def total_cost(assignment, cost_matrix):
    total = 0
    for i in range(len(assignment)):
        total += cost_matrix[i][assignment[i]]
    return total
import itertools
def assignment_problem(cost_matrix):
    n = len(cost_matrix)
    workers = list(range(n))  
    permutations = itertools.permutations(workers)
    min_cost = float('inf')
    optimal_assignment = None
    for perm in permutations:
        cost = total_cost(perm, cost_matrix)
        if cost < min_cost:
            min_cost = cost
            optimal_assignment = perm    
    return min_cost, optimal_assignment
cost_matrix = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
]

min_cost, optimal_assignment = assignment_problem(cost_matrix)

print(f"Minimum Cost: {min_cost}")
print(f"Optimal Assignment: {optimal_assignment}")

#14 0/1 knapsack
def knapsack(values, weights, W):
    n = len(values)
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]
values = [60, 100, 120]
weights = [10, 20, 30]
W = 50
print(knapsack(values, weights, W))

