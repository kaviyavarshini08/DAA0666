
#3
N = 9
def printing(arr):
	for i in range(N):
		for j in range(N):
			print(arr[i][j], end = " ")
		print()
def isSafe(grid, row, col, num):
	for x in range(9):
		if grid[row][x] == num:
			return False
	for x in range(9):
		if grid[x][col] == num:
			return False
	startRow = row - row % 3
	startCol = col - col % 3
	for i in range(3):
		for j in range(3):
			if grid[i + startRow][j + startCol] == num:
				return False
	return True
def solveSudoku(grid, row, col):
	if (row == N - 1 and col == N):
		return True
	if col == N:
		row += 1
		col = 0
	if grid[row][col] > 0:
		return solveSudoku(grid, row, col + 1)
	for num in range(1, N + 1, 1):
		if isSafe(grid, row, col, num):
			grid[row][col] = num
			if solveSudoku(grid, row, col + 1):
				return True
		grid[row][col] = 0
	return False
grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
		[5, 2, 0, 0, 0, 0, 0, 0, 0],
		[0, 8, 7, 0, 0, 0, 0, 3, 1],
		[0, 0, 3, 0, 1, 0, 0, 8, 0],
		[9, 0, 0, 8, 6, 3, 0, 0, 5],
		[0, 5, 0, 0, 9, 0, 6, 0, 0],
		[1, 3, 0, 0, 0, 0, 2, 5, 0],
		[0, 0, 0, 0, 0, 0, 0, 7, 4],
		[0, 0, 5, 2, 0, 6, 3, 0, 0]]

if (solveSudoku(grid, 0, 0)):
	printing(grid)
else:
	print("no solution exists ")

    
#5
def findTargetSumWays(nums, target):
    from collections import defaultdict
    dp = defaultdict(int)
    dp[0] = 1
    
    for num in nums:
        next_dp = defaultdict(int)
        for s in dp:
            next_dp[s + num] += dp[s]
            next_dp[s - num] += dp[s]
        dp = next_dp
    
    return dp[target]

# Example usage:
nums = [1, 1, 1, 1, 1]
target = 3
print(findTargetSumWays(nums, target))  # Output: 5

#6
def sumSubarrayMins(arr):
    mod = 10**9 + 7
    n = len(arr)
    
    # Previous Less Element
    ple = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        if stack:
            ple[i] = stack[-1]
        stack.append(i)
    
    # Next Less Element
    nle = [n] * n
    stack = []
    for i in range(n-1, -1, -1):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        if stack:
            nle[i] = stack[-1]
        stack.append(i)
    
    # Calculate result
    result = 0
    for i in range(n):
        left = i - ple[i]
        right = nle[i] - i
        result = (result + arr[i] * left * right) % mod
    
    return result

# Example usage:
arr = [3, 1, 2, 4]
print(sumSubarrayMins(arr))  # Output: 17

#7
def combinationSum(candidates, target):
    result = []
    
    def backtrack(start, current_combination, current_sum):
        if current_sum == target:
            result.append(list(current_combination))
            return
        if current_sum > target:
            return
        
        for i in range(start, len(candidates)):
            current_combination.append(candidates[i])
            backtrack(i, current_combination, current_sum + candidates[i])
            current_combination.pop()
    
    backtrack(0, [], 0)
    return result

# Example usage:
candidates = [2, 3, 6, 7]
target = 7
print(combinationSum(candidates, target))  # Output: [[2, 2, 3], [7]]

candidates = [2, 3, 5]
target = 8
print(combinationSum(candidates, target))  # Output: [[2, 2, 2, 2], [2, 3, 3], [3, 5]]

#8
def combinationSum2(candidates, target):
    candidates.sort()
    result = []
    combination = []
    
    def backtrack(start, target):
        if target == 0:
            result.append(combination[:])
            return
        if target < 0:
            return
        
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            combination.append(candidates[i])
            backtrack(i + 1, target - candidates[i])
            combination.pop()
    
    backtrack(0, target)
    return result

# Example 1
candidates = [10, 1, 2, 7, 6, 1, 5]
target = 8
print(combinationSum2(candidates, target))
# Output: [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]

# Example 2
candidates = [2, 5, 2, 1, 2]
target = 5
print(combinationSum2(candidates, target))
# Output: [[1, 2, 2], [5]]

#9
def permute(nums):
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result

# Example 1
nums = [1, 2, 3]
print(permute(nums))
# Output: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

# Example 2
nums = [0, 1]
print(permute(nums))
# Output: [[0, 1], [1, 0]]

# Example 3
nums = [1]
print(permute(nums))
# Output: [[1]]

#10
def permuteUnique(nums):
    nums.sort()
    result = []
    used = [False] * len(nums)
    
    def backtrack(combination):
        if len(combination) == len(nums):
            result.append(combination[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            used[i] = True
            combination.append(nums[i])
            backtrack(combination)
            combination.pop()
            used[i] = False
    
    backtrack([])
    return result

# Example 1
nums = [1, 1, 2]
print(permuteUnique(nums))
# Output: [[1, 1, 2], [1, 2, 1], [2, 1, 1]]

# Example 2
nums = [1, 2, 3]
print(permuteUnique(nums))
# Output: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

#11 and #12
def greedy_coloring(edges, n, k):
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Colors array, initialized to -1 (uncolored)
    colors = [-1] * n
    
    # The result counts how many regions you color before all are colored
    result = 0

    def is_valid_color(node, color):
        for neighbor in adj_list[node]:
            if colors[neighbor] == color:
                return False
        return True

    # Function to color the graph using the greedy algorithm
    def color_graph(node):
        nonlocal result
        if node == n:
            return True

        for color in range(k):
            if is_valid_color(node, color):
                colors[node] = color
                result += 1
                if color_graph(node + 1):
                    return True
                # Backtrack
                colors[node] = -1
                result -= 1
        
        return False

    color_graph(0)
    return result

# Example usage:
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
n = 4
k = 3
print(greedy_coloring(edges, n, k))  # Output will depend on the greedy algorithm's execution

#13
def has_hamiltonian_cycle(edges, n):
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    def backtrack(path):
        if len(path) == n:
            return path[0] in adj_list[path[-1]]
        
        for neighbor in adj_list[path[-1]]:
            if neighbor not in path:
                path.append(neighbor)
                if backtrack(path):
                    return True
                path.pop()
        
        return False

    for start in range(n):
        if backtrack([start]):
            return True

    return False

# Example usage:
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 4), (4, 0)]
n = 5
print(has_hamiltonian_cycle(edges, n))  # Output: True

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
n = 4
print(has_hamiltonian_cycle(edges, n))  # Output: True

#14
def has_hamiltonian_cycle(edges, n):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    def backtrack(path, visited):
        if len(path) == n:
            # Check if there is an edge from the last vertex in path to the first vertex
            if path[-1] in graph[path[0]]:
                return True
            else:
                return False
        
        last_node = path[-1]
        for neighbor in graph[last_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                if backtrack(path, visited):
                    return True
                path.pop()
                visited.remove(neighbor)
        
        return False
    
    # Try to find a Hamiltonian cycle starting from each vertex
    for start in range(n):
        if backtrack([start], {start}):
            return True
    
    return False

# Example usage:
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
n = 4
print(has_hamiltonian_cycle(edges, n))  # Output: True

#15
def subsets_lexicographical_order(S):
    S.sort()
    result = []
    
    def backtrack(start, current):
        result.append(current[:])
        
        for i in range(start, len(S)):
            current.append(S[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result

# Example usage:
A = [1, 2, 3]
print(subsets_lexicographical_order(A))
# Output: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]

#16
def subsets_with_element(E, x):
    result = []
    
    def backtrack(start, current):
        if x in current:
            result.append(current[:])
        
        for i in range(start, len(E)):
            current.append(E[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result

# Example usage:
E = [2, 3, 4, 5]
x = 3
print(subsets_with_element(E, x))
# Output: [[3], [2, 3], [3, 4], [3, 5], [2, 3, 4], [2, 3, 5], [3, 4, 5], [2, 3, 4, 5]]

#17
def universal_strings(words1, words2):
    result = []
    set_words2 = [set(word) for word in words2]
    
    for word1 in words1:
        is_universal = True
        set_word1 = set(word1)
        for set_word2 in set_words2:
            if not set_word2.issubset(set_word1):
                is_universal = False
                break
        if is_universal:
            result.append(word1)
    
    return result

# Example usage:
words1 = ["amazon", "apple", "facebook", "google", "leetcode"]
words2 = ["e", "o"]
print(universal_strings(words1, words2))
# Output: ["facebook", "google", "leetcode"]

words2 = ["l", "e"]
print(universal_strings(words1, words2))
# Output: ["apple", "google", "leetcode"]


