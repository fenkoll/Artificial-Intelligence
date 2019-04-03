# -*- coding: utf-8 -*-
import numpy as np
import copy as cp
import queue

import numpy as np
import time
import math
# from collections import defaultdict
# used from from https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html with changes
def get_pent_idx(pent):
    """
    Returns the index of a pentomino.
    """
    pidx = 0
    for i in range(pent.shape[0]):
        for j in range(pent.shape[1]):
            if pent[i][j] != 0:
                pidx = pent[i][j]
                break
        if pidx != 0:
            break
    if pidx == 0:
        return -1
    return pidx - 1

global pent_dic
def get_origin_pent(pent, pents):
    if (pent not in pent_dic):
        pent_dic = {}
        for i in pents:
            pent_dic[get_pent_idx(i)] = i
    idx = get_pent_idx(pent)
    return pent_dic(idx)

def solve_x(X, Y, solution=[]):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve_x(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)

def add_pentomino(board, pent, coord):
    """
    Adds a pentomino pent to the board. The pentomino will be placed such that
    coord[0] is the lowest row index of the pent and coord[1] is the lowest
    column index.

    check_pent will also check if the pentomino is part of the valid pentominos.
    """
    for row in range(pent.shape[0]):
        for col in range(pent.shape[1]):
            if pent[row][col] != 0:
                if board[coord[0]+row][coord[1]+col] != 0: # Overlap
                    return False
                else:
                    board[coord[0]+row][coord[1]+col] = pent[row][col]
    return True
def check_pentomino(board, pent, coord):
    """
    Adds a pentomino pent to the board. The pentomino will be placed such that
    coord[0] is the lowest row index of the pent and coord[1] is the lowest
    column index.

    check_pent will also check if the pentomino is part of the valid pentominos.
    """

    for row in range(pent.shape[0]):
        for col in range(pent.shape[1]):
            if pent[row][col] != 0:
                if coord[0] + row >= len(board) or coord[1] + col >= len(board[0]) or board[coord[0]+row][coord[1]+col] != 0: # Overlap
                    return False
    return True
def eva(board):
    res = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            t = 0
            if board[i][j] == 0 and i + 1 == len(board):
                t += 1
            if board[i][j] == 0 and j + 1 == len(board[0]):
                t += 1
            if board[i][j] == 0 and i == 0:
                t += 1
            if board[i][j] == 0 and j == 0:
                t += 1
            if board[i][j] == 0 and i + 1 < len(board) and board[i + 1][j] > 0:
                t += 1
            if board[i][j] == 0 and i > 0 and board[i - 1][j] > 0:
                t += 1
            if board[i][j] == 0 and j + 1 < len(board[0]) and board[i][j + 1] > 0:
                t += 1
            if board[i][j] == 0 and j > 0 and board[i][j - 1] > 0:
                t += 1
            if t == 4:
                t += 100000
            # print(t)
            res += t
    return float(res)
global pent_id_dic
pent_id_dic = {}
def pent_id(pent):
    res = 0
    for i in range(5):
        for j in range(5):
            if i < len(pent) and j < len(pent[0]) and pent[i][j]:
                # print(i, j, 2 ** (5 * i + j))
                res +=  2 ** (5 * i + j)

    pent_id_dic[res] = pent
    return res

def pent_hash(pent, i, j, board):
    k = pent_id(pent)
    return k * len(board) * len(board[0]) + i * len(board[0]) + j

def get_pent_from_hashed(k, board):

    # print(k / (len(board) * len(board[0])))
    id = math.floor(k / (len(board) * len(board[0])))
    t = k - id * len(board) * len(board[0])
    x = math.floor(t / len(board[0]))
    y = t - x * (len(board[0]))
    # print(pent_id_dic)
    return (pent_id_dic[id], (x, y))

def get_all_subsets(board, all_pents, pents):
    res = {}
    for p in all_pents:
        for i in range(len(board) - len(p) + 1):
            for j in range(len(board[0]) - len(p[0]) + 1):
                if (check_pentomino(board, p, (i, j))):
                    # t = pent_hash(p, i, j, board)
                    # ll = get_pent_from_hashed(t, board)
                    temp = []
                    temp.append(get_pent_idx(p))
                    for r in range(len(p)):
                        for s in range(len(p[0])):
                            if p[r][s]:
                                temp.append(len(pents) + len(board[0]) * (r + i) + j + s)
                    res[pent_hash(p, i, j, board)] = temp
    return res



def solve(board1, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is
    the coordinate of the upper left corner of pi in the board (lowest row and column index
    that the tile covers).

    -Use np.flip and np.rot90 to manipulate pentominos.

    -You can assume there will always be a solution.
    """


    board = 1 - board1
    # print(board)
    # print(pents[0])
    # print(pent_id(pents[0]))
    # print(pents[0][0][0])
    # return
    all_pents = []
    temp = cp.deepcopy(pents)
    # print(get_pent_from_hashed(pent_hash(pents[1], 2, 2, board), board))
    # return
    for i in range(4):
        for p in pents:
            all_pents.append(np.flipud(np.rot90(p, i)))
            all_pents.append(np.rot90(p, i))
    # print(len(all_pents))
    # for i in all_pents:
    #     if (get_pent_idx(i) == 2):
    #         print(i)
    repeat = []
    for i in range(len(all_pents)):
        for j in range(len(all_pents)):
            if i < j and np.array_equal(all_pents[i], all_pents[j]):
                repeat.append(j)
    repeat = list(set(repeat))
    repeat.sort()
    repeat.reverse()
    for i in repeat:
        all_pents.pop(i)
    for i in all_pents:
        pent_id(i)
    y = get_all_subsets(board, all_pents, pents)
    x = range(len(pents) + len(board) * len(board[0]))
    x2 = {j: set(filter(lambda i: j in y[i], y)) for j in x}
    # print(len(y))
    # print(len(x))
    solutions = solve_x(x2, y)
    # print(sol)
    # for i in sol:
    #     print(get_pent_from_hashed(i, board))
    # print(len(all_pents))
    # for j in range(1, 13):
    #     for i in all_pents:
    #         if (get_pent_idx(i) == j):
    #             print(i)

    # print(sum)
    for i in solutions:
        sol = i
        break
    # print(sol)
    res = [get_pent_from_hashed(r, board) for r in sol]
    b = cp.deepcopy(board)

    for i in res:
        add_pentomino(b, i[0], i[1])
    # print(b)
    return res
