# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)
import copy as cp
import queue
def heuristics(maze,point):
    object = maze.getObjectives()
    distance = abs(object[0][0]-point[0])+abs(object[0][1]-point[1])
    return distance

class Path:
    def __init__(self, l, t, m):
        # type is 0 when it's used for heuristic, it's 1 for total cost of A star search
        self.list = l
        self.type = t
        self.maze = m
    def heuristics(self):
        object = self.maze.getObjectives()
        point = self.list[len(self.list) - 1]
        distance = abs(object[0][0]-point[0])+abs(object[0][1]-point[1])
        return distance
    def c(self):
        return len(self.list) - 1 + self.heuristics() * 0.99
    def __lt__(self, other):
        if (self.type):
            return self.c() < other.c()
        return self.heuristics() < self.heuristics()
    def __le__(self, other):
        if (self.type):
            return self.c() <= other.c()
        return self.heuristics() <= self.heuristics()
    def __gt__(self, other):
        if (self.type):
            return self.c() > other.c()
        return self.heuristics() > self.heuristics()
    def __ge__(self, other):
        if (self.type):
            return self.c() >= other.c()
        return self.heuristics() >= self.heuristics()
    def __eq__(self, other):
        if (self.type):
            return self.c() == other.c()
        return self.heuristics() == self.heuristics()
    def __ne__(self, other):
        if (self.type):
            return self.c() != other.c()
        return self.heuristics() != self.heuristics()
    def next(self, p):
        temp = cp.deepcopy(self.list)
        temp.append(p)
        return Path(temp, self.type, self.maze)

    def get_path(self):
        return self.list



def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)
def bfs_single(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    startpoint = maze.getStart()
    flag = False
    Q = queue.Queue()
    Q.put(startpoint)
    ans = []
    ans_count = 0
    dimenstion = maze.getDimensions()
    path = {}
    path[startpoint] = startpoint
    visit = [([False]*dimenstion[1])for i in range (dimenstion[0])]
    # direction = [[1,0],[-1,0],[0,1],[0,-1]]
    visit[startpoint[0]][startpoint[1]]= True
    object = maze.getObjectives()
    while(not Q.empty()):
        vn = Q.get()
        vn_neighbor = maze.getNeighbors(vn[0],vn[1])
        for each_point in vn_neighbor:
            if(maze.isObjective(each_point[0],each_point[1])):
                path[each_point] = vn
                cur = each_point
                flag = True
                break
            elif(visit[each_point[0]][each_point[1]]==False):
                path[each_point] = vn
                Q.put(each_point)
                visit[each_point[0]][each_point[1]] = True
        if flag == True:
            break
    while(path[cur]!=startpoint):
        ans.append(cur)
        cur = path[cur]
    ans.append(cur)
    ans.append(startpoint)
    for each_point1 in visit:
        for each_point2 in each_point1:
            if each_point2 == True:
                ans_count+=1

    return ans, ans_count

def greedy_single(maze):
    ans = []
    ans_count = 0
    startpoint = maze.getStart()
    Q = queue.PriorityQueue()
    Q.put((0,startpoint))
    path = {}
    path[startpoint] = startpoint
    dimenstion = maze.getDimensions()
    visit = [([False]*dimenstion[1])for i in range (dimenstion[0])]
    # direction = [[1,0],[-1,0],[0,1],[0,-1]]
    visit[startpoint[0]][startpoint[1]]= True
    while(not Q.empty()):
        vn = Q.get()
        vn_neighbor = maze.getNeighbors(vn[1][0],vn[1][1])
        for each_point in vn_neighbor:
            if maze.isObjective(each_point[0],each_point[1]):
                path[each_point] = vn[1]
                cur = each_point
                break
            elif(visit[each_point[0]][each_point[1]]==False):
                Q.put((heuristics(maze,each_point),each_point))
                path[each_point] = vn[1]
                visit[each_point[0]][each_point[1]] = True
    # retur path, num_states_explored
    while(path[cur]!=startpoint):
        ans.append(cur)
        cur = path[cur]
        ans_count += 1
    return ans, ans_count

def get_map(maze):
    m = maze.getDimensions()[0]
    n = maze.getDimensions()[1]
    res = [[0 for i in range(m)] for j in range(n)]
    for i in range(m):
        for j in range(n):
            if maze.isObjective(i, j):
                res[i, j] = 100
            if maze.isWall(i, j):
                res[i, j] = -1
    return res

def dfs_single(maze):
    path=[]
    num_states_explored=0
    visited_states=[]
    start = maze.getStart()
    objectives = maze.getObjectives()
    frontier_list=[[start]]

    while frontier_list:
        path=frontier_list.pop()
        cur_state=path[-1]
        if (cur_state[0], cur_state[1]) in objectives:
            break
        if cur_state not in visited_states:
            visited_states.append(cur_state)
            num_states_explored+=1

        neighbor_states=maze.getNeighbors(cur_state[0], cur_state[1])
        for each in neighbor_states:
            if each not in visited_states and maze.isValidMove(each[0],each[1]):
                temp = list(path)
                temp.append(each)
                frontier_list.append(temp)

    return path, num_states_explored

def astar_single(maze):
    a,b = bfs(maze)
    a = a[::-1]
    ct = 0
    sp = maze.getStart()
    q = queue.PriorityQueue()
    first = Path([sp], 1, maze)
    q.put(first)
    dimenstion = maze.getDimensions()
    visit = [([False]*dimenstion[1])for i in range (dimenstion[0])]
    # direction = [[1,0],[-1,0],[0,1],[0,-1]]
    ct = 1;
    visit[sp[0]][sp[1]]= True
    while(not q.empty()):
        t = q.get()
        l = t.get_path()
        t_neighbor = maze.getNeighbors(l[len(l) - 1][0],l[len(l) - 1 ][1])
        for p in t_neighbor:
            if maze.isObjective(p[0],p[1]):
                l.append(p)
                return l, ct
            elif(visit[p[0]][p[1]]==False):
                ct+=1;
                nex = t.next(p)
                q.put(nex)
                visit[p[0]][p[1]] = True
    # retur path, num_states_explored


def bfs(maze):
    if len(maze.getObjectives()) == 1:
        return bfs_single(maze)
    # TODO: Write your code here
    # return path, num_states_explored
    return multiple_dots_handler(maze, bfs_single)
    return [], 0


def dfs(maze):
    if len(maze.getObjectives()) == 1:
        return dfs_single(maze)
    # TODO: Write your code here
    # return path, num_states_explored
    return multiple_dots_handler(maze, dfs_single)



def greedy(maze):
    if len(maze.getObjectives()) == 1:
        return greedy_single(maze)
    # TODO: Write your code here
    # return path, num_states_explored
    return multiple_dots_handler(maze, greedy_single)



def astar(maze):
    if len(maze.getObjectives()) == 1:
        return astar_single(maze)
    # TODO: Write your code here
    # return path, num_states_explored
    return multiple_dots_handler(maze, astar_single)

def multiple_dots_handler(maze, method):
    cur = cp.deepcopy(maze)
    obs = cur.getObjectives()
    ct = 0
    startp = cur.getStart()
    res = []
    while len(obs) > 0:
        min = obs[0]
        minlen = abs(obs[0][0] - startp[0]) + abs(obs[0][1] - startp[1])
        for i in obs:
            if abs(i[0] - startp[0]) + abs(i[1] - startp[1]) < minlen:
                minlen = abs(i[0] - startp[0]) + abs(i[1] - startp[1])
                min = i
        if min in res:
            obs.remove(min)
            cur.setObjectives(obs)
        else:
            temp = cp.deepcopy(cur)
            temp.setObjectives([min])
            res += method(temp)[0]
            ct +=  method(temp)[1]
            obs.remove(min)
            cur.setObjectives(obs)
            cur.setStart(min)
    return res, ct
