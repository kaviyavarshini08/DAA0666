#1 first pali
def pali(str):
    for i in s:
        if i==i[::-1]:
            return i
    return ""
s=input().split()
print(pali(s))

#2 n1 in n2, n2 in n1
def n(n1,n2):
    sn1=set(n1)
    sn2=set(n2)
    a1=sum(1 for i in n1 if i in sn2)
    a2=sum(1 for i in n2 if i in sn1)
    return a1,a2
n1=list(map(int,input().split()))
n2=list(map(int,input().split()))
print(n(n1,n2))

#3 sq of len val
def sum(nums):
    n = len(nums)
    t= 0
    for i in range(n):
        for j in range(i, n):
           arr = nums[i:j+1]
           dis = len(set(arr))
           t+= dis ** 2    
    return t
nums = [1, 2, 1]
print(sum(nums))

#4 indices//k
def sub(n,k):
    m=len(n)
    c=0
    for i in range (m):
        for j in range(i+1,m):
            if n[i]==n[j]:
                s=i*j
                if s%k==0:
                    c+=1
    return c
n=list(map(int,input().split()))
k= int(input())
print(sub(n,k))

#5 sort max
n=list(map(int,input().split()))
m=sorted(n)
print(m[-1])

#6 sort return none or max
def sort(n):
    if n==[]:
        return None
    for i in range (len(n)):
        for j in range(i,len(n)):
            if( n[i]>n[j]):
                n[i],n[j]=n[j],n[i]
    return n[-1], n
n=list(map(int,input().split()))
print(sort(n))

#7 rem dup
def unique(lst):
    s=[]
    u=[]
    for i in lst:
        if i not in s:
            u.append(i)
            s.append(i)
    return u
lst=list(map(int,input().split()))
print(unique(lst))

#8 bubble sort
def bsort(arr):
    for i in range (len(arr)):
        for j in range(0,len(arr)-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr
arr=list(map(int,input().split()))
print(bsort(arr))

#9 bin search
def bs(arr,x):
    sarr=sorted(arr)
    l,r=0,len(sarr)-1
    while( l<=r):
        m=(l+r)//2
        if sarr[m]==x:
            return True
        elif sarr[m]<x:
            l=m+1
        else:
            r=m-1
    return False
arr=list(map(int,input().split()))
x=int(input())
print(bs(arr,x))

#10 sort nlogn
def quick(arr):
    if len(arr)<1:
        return arr
    p=arr[len(arr)//2]
    l=[x for x in arr if x<p]
    m=[x for x in arr if x==p]
    r=[x for x in arr if x>p]
    return quick(l)+m+quick(r)
arr=list(map(int,input().split()))
print(quick(arr))

#11 out of grid
def number(m, n, N, i, j):
    dict = {}
    if i < 0 or i >= m or j < 0 or j >= n:
        return 1
    if N == 0:
        return 0
    if (N, i, j) in dict:
        return dict[(steps, i, j)]   
    ways = 0
    for i in range (m*n):
         ways += number(m, n, N - 1, i - 1, j)    
         dict[(N, i, j)] = ways
    return ways
m = 2
n = 2
N = 2
srow = 0
scol = 0
ways = number(m, n, N, srow, scol)
print("Number of ways to move the ball out of the grid in exactly N steps:", ways)

#12 robber
def rob(nums):
    def rob1(h):
        p1 = p2 = 0
        for num in nums:
             p1, p2 = max(num + p2, p1), p1
        return p1
    if len(nums) == 1:
        return nums[0]    
    return max(rob(nums[:-1]), rob(nums[1:]))
print(rob([1,2,3,1]))  

#13 stairs 
def stairs(n):
    if n == 0 or n == 1:
        return 1    
    p1, p2 = 1, 1
    for i in range(2, n + 1):
        current = p1 + p2
        p2 = p1
        p1 = current    
    return p1
print(stairs(4)) 
print(stairs(3))  

#14 path by robot
def fact(a):
    f = 1
    for i in range(1,a+1):
        f = f*i
    return f
m = 3
n = 2
a = (m+n-2)-(m-1)
b = m+n-2
c = m-1
print(fact(b)//(fact(c)*fact(a)))

#15 large str
def large(s):
    result, start = [], 0
    for end in range(1, len(s) + 1):
        if end == len(s) or s[end] != s[start]:
            if end - start >= 3:
                result.append([start, end - 1])
            start = end
    return result
s = "abbxxxxzzy"
print(large(s)) 

#16 grid
def game(board):
    def coun(board, r, c):
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),         (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        liveneighbors = 0
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            if 0 <= rr < len(board) and 0 <= cc < len(board[0]) and abs(board[rr][cc]) == 1:
                liveneighbors += 1
        return liveneighbors
    rows, cols = len(board), len(board[0])
    nextstate = [[0] * cols for _ in range(rows)]    
    for r in range(rows):
        for c in range(cols):
            liveneighbors = coun(board, r, c)           
            if board[r][c] == 1:
                if liveneighbors < 2 or liveneighbors > 3:
                    nextstate[r][c] = 0  
                else:
                    nextstate[r][c] = 1  
            else:
                if liveneighbors == 3:
                    nextstate[r][c] = 1
                else:
                    nextstate[r][c] = 0 
    for r in range(rows):
        for c in range(cols):
            board[r][c] = nextstate[r][c]
board1 = [
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 0]
]
game(board1)
print(board1)

#17 towe of glass
def tower(poured, row, glass):
    dp = [[0.0] * (r + 1) for r in range(row + 1)]
    dp[0][0] = poured  
    for i in range(row):
        for j in range(len(dp[i])):
            excess = (dp[i][j] - 1.0) / 2.0
            if excess > 0:
                dp[i + 1][j] += excess
                dp[i + 1][j + 1] += excess    
    return dp[row][glass]
poured = 2
row = 1
glass = 1
print(tower(poured, row, glass)) 
