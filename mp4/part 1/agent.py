import numpy as np
import utils
import random
def wall_check(head):
    if head == 12:
        return 2
    if head == 1:
        return 1
    return 0

def food_check(food, head):
    if food > head:
        return 2
    elif food < head:
        return 1
    return 0

def body_check(head_x, head_y, body):
    return [int((head_x + a[0], head_y + a[1]) in body) \
    for a in ((0,1),(0,-1),(-1,0),(1,0))]

def f(u, n, ne):
    if n < ne:
        return 1
    else:
        return u
def discretize(state):
    head_x = state[0] // 40
    head_y = state[1] // 40
    body = [(a[0] // 40, a[1] // 40)  for a in state[2]]
    food_x = state[3] // 40
    food_y = state[4] // 40
    r = body_check(head_x, head_y, body)
    return wall_check(head_x), wall_check(head_y), \
    food_check(food_x, head_x), food_check(food_y, head_y),\
    r[0], r[1], r[2], r[3]

class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.points = 0
        self.s = None
        self.a = None

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''

        d1 = discretize(state)
        q1 = self.Q[d1[0], d1[1], d1[2], d1[3], d1[4], d1[5], d1[6], d1[7]]
        # print(state)

        if self.s:
            q2 = self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7]]
            n2 = self.N[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7]]
            r = -0.1
            if points > self.points:
                r = 1
            elif dead:
                r = -1

            alpha = self.C/(self.C + n2[self.a])
            q2[self.a] = q2[self.a] + alpha * (r + self.gamma * np.max(q1) - q2[self.a])

            # def train_more(a):
            #     r = 0
            #     if a == 3 and (d1[0] == 2 or d1[7]):
            #         r = -1
            #     if a == 2 and (d1[0] == 1 or d1[6]):
            #         r = -1
            #     if a == 1 and (d1[1] == 2 or d1[5]):
            #         r = -1
            #     if a == 0 and (d1[1] == 1 or d1[4]):
            #         r = -1
            #     alpha = self.C/(self.C + n2[a])
            #     q2[a] = q2[a] + alpha * (r + self.gamma * np.max(q1) - q2[a])
            # for i in range(4):
            #     if i != self.a:
            #         train_more(i)

        max = -10000000000
        res = -1
        n1 = self.N[d1[0], d1[1], d1[2], d1[3], d1[4], d1[5], d1[6], d1[7]]
        for i in range(4):
            if f(q1[i], n1[i], self.Ne) >= max:
                max = f(q1[i], n1[i], self.Ne)
                res = i
        if not dead:
            n1[res] += 1
            self.s = d1
            self.a = res
            self.points = points
        else:
            self.reset()
        # print(res)
        return res
