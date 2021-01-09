#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thus Jan 07 14:44:12 2021
@author: Filipe Souza

Based on Josh Varty (https://github.com/JoshVarty/AlphaZeroSimple)
"""
import gym
import numpy as np
from model import CNNmodel
from trainer import Trainer
from tkinter import Tk,Label,mainloop

def getAction(action):
    if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, np.ndarray):
        action = args['boardSize'] * action[0] + action[1]
    elif action is None:
        action = args['boardSize'] ** 2
    return action

def alert_popup(title, message, path):
    root = Tk()
    root.title(title)
    w = 200     # popup window width
    h = 100     # popup window height
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w)/2
    y = (sh - h)/2
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    m = message
    m += '\n'
    m += path
    w = Label(root, text=m, width=120, height=10)
    w.pack()
    mainloop()

args = {
    'boardSize': 9,
     1.0: 'BLACK', 
    -1.0: 'WHITE',
    'batchSize': 64,
    'numIters': 500,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 100,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 50,                                    # Number of epochs of training per iteration
    'checkpointPath': 'model.hdf5'                  # location to save latest set of weights
}

game = None #gym.make('gym_go:go-v0', size=args['boardSize'], komi=0, reward_method='heuristic')

model = CNNmodel(args['boardSize'], (args['boardSize'] * args['boardSize'])+1, args)

trainer = Trainer(game, model, args)
trainer.learn()


game = gym.make('gym_go:go-v0', size=args['boardSize'], komi=0, reward_method='heuristic')

while not(game.done):
    actions, value = model.predict(game.state())
    valid_moves = game.valid_moves()
    action_probs = actions * valid_moves  # mask invalid moves
    action_probs /= np.sum(action_probs)
    action = (args['boardSize'] * args['boardSize'])
    if np.argmax(action_probs[:-1]) > 0:
        action = np.argmax(action_probs[:-1])
    game.step(action)
    if not(game.done):
        validMove = game.valid_moves()
        action = game.render('human')
        while validMove[getAction(action)]==0:
            action = game.render('human')
        game.step(getAction(action))
    
alert_popup("!!!Winner!!!", "The winner is:", args[game.winner()])


