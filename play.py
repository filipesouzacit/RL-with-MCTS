#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thus Jan 07 15:43:21 2021

@author: Filipe Souza
"""
import gym
import numpy as np
from model import CNNmodel
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
    'checkpointPath': 'model9.hdf5'
}


model = CNNmodel(args['boardSize'], (args['boardSize'] * args['boardSize'])+1, args)
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


