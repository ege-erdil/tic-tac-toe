import torch
import numpy as np
from copy import deepcopy

class TicTacToe:
    def __init__(self, boardV=torch.zeros(size=(3, 3)), currPlayerV=1):
        self.board = boardV
        self.currPlayer = currPlayerV

    def currBoard(self):
        return tuple([x.item() for x in torch.flatten(self.board)])

    def move(self, pos):
        if self.board[pos] != 0:
            return -self.currPlayer
        
        self.board[pos] = self.currPlayer
        r = self.winner()

        self.currPlayer = -self.currPlayer
        return r

    def winner(self):
        for p in [-1, 1]:
            for j in range(3):
                if torch.all(torch.eq(self.board[j], p)):
                    return p
                if torch.all(torch.eq(self.board[:, j], p)):
                    return p

            if self.board[0, 0] == p and self.board[1, 1] == p and self.board[2, 2] == p:
                return p

            if self.board[2, 0] == p and self.board[1, 1] == p and self.board[0, 2] == p:
                return p

        if torch.all(torch.logical_not(torch.eq(self.board, 0))):
            return 0

        return None

def tup_to_board(tup):
    return torch.reshape(torch.FloatTensor([[tup[3*i + j] for j in range(3)] for i in range(3)]), (1, 3, 3))

V = torch.nn.Sequential(
        torch.nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride=1, padding=0),
        torch.nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride=1, padding=0),
        torch.nn.Conv2d(10, 1, kernel_size=1, stride=1, padding=0),
        torch.nn.Flatten(),
        torch.nn.Tanh()
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(V.parameters(), lr=1e-3)

empty = TicTacToe(torch.zeros(size=(3, 3)))

cache = dict()
states = []

moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

epsilon = 0.1

def in_cache(pair):
    for e in cache:
        if torch.all(torch.eq(pair[0], e[0])) and e[1] == pair[1]:
            return True
    return False
    
for t in range(3000):
    bState = TicTacToe(torch.zeros(size=(3, 3)))
    remove_keys = []
    
    if t % 100 == 99:
        for key in cache:
            if cache[key][1] + 400 < t:
                remove_keys.append(key)
        print("Removed length: %d" % len(remove_keys))
    
    for key in remove_keys:
        states.remove(key)
        cache.pop(key)
    
    while True:
        p = bState.currPlayer
        optValue = p * (-float("inf"))
        avgVal = 0
        for m in moves:
            state = deepcopy(bState)
            reward = state.move(m)

            if reward == None:
                value = V(torch.reshape(state.board, (1, 3, 3)))
            else:
                value = reward

            avgVal += value

            if (p == 1 and optValue < value) or (p == -1 and optValue > value):
                optValue = value
                optReward = reward
                optMove = m

        avgVal /= len(moves)
        cache[bState.currBoard()] = ((1 - epsilon) * optValue + epsilon * avgVal, t)
        if bState.currBoard() not in states:
            states.append(bState.currBoard())

        if np.random.rand() > epsilon:   
            bState.move(optMove)
        else:
            bState.move(moves[np.random.choice(range(len(moves)))])
            
        if optReward != None:
            break
            
    indices = range(len(states))
    batch = [states[i] for i in indices]
    
    y = V(torch.stack([tup_to_board(q) for q in batch]))
    y_target = [cache[q][0] for q in batch]

    y_target = torch.FloatTensor(y_target)
    loss = loss_fn(torch.flatten(y), y_target)

    if t % 100 == 99:
        print(t, float(loss), float(V(torch.reshape(empty.board, (1, 3, 3)))), len(cache))


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

playState = TicTacToe(torch.zeros(size=(3, 3)))

def playerMove(m):
    playState.move(m)
    print(playState.board)

def computerMove():
    p = playState.currPlayer
    optValue = p * (-float("inf"))
    for m in moves:
        state = deepcopy(playState)
        reward = state.move(m)

        if reward == None:
            value = V(torch.reshape(state.board, (1, 3, 3)))
        else:
            value = reward

        print(m, value, reward)

        if (p == 1 and optValue < value) or (p == -1 and optValue > value):
            optValue = value
            optReward = reward
            optMove = m
    
    playState.move(optMove)
    print(playState.board, optValue)
