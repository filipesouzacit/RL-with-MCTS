#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thus Jan 07 15:54:13 2021
@author: Filipe Souza

Based on Josh Varty (https://github.com/JoshVarty/AlphaZeroSimple)
"""
import numpy as np
from random import shuffle
import keras

from gym_go import gogame
from monte_carlo_tree_search import MCTS

class Trainer:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)

    def exceute_episode(self):

        train_examples = []
        current_player = 1
        state = gogame.init_state(self.args['boardSize'])

        while True:
            #print("while True")
            canonical_board = gogame.canonical_form(state)

            self.mcts = MCTS(self.game, self.model, self.args)
            root = self.mcts.run(self.model, canonical_board, to_play=1)

            action_probs = [0 for _ in range((self.args['boardSize']* self.args['boardSize'])+1)]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((canonical_board, current_player, action_probs))

            action = root.select_action(temperature=1)
            state = gogame.next_state(state, action, canonical=False)
            current_player = - current_player
            reward = gogame.winning(state)*current_player if gogame.game_ended(state) else None 

            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    tfBoard = np.array([hist_state[0],hist_state[1],hist_state[3]]).transpose().tolist()
                    #ret.append(np.array([tfBoard,tfBoard, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))]))
                    ret.append((tfBoard,hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))
                return ret

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):

            print("numIters: {}/{}".format(i, self.args['numIters']))

            train_examples = []

            for eps in range(self.args['numEps']):
                print("numEps: {}/{}".format(eps, self.args['numEps']))
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)

    def train(self, trainD):
        
        # Define the checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(self.args['checkpointPath'], monitor="val_loss",
                                                  mode="min", save_best_only=True, verbose=0)

        # train the network
        print("Training network...")
        
        x = [i[0] for i in trainD]
        x = np.array(x)
        
        y1 = [i[1] for i in trainD]
        y2 = [i[2] for i in trainD]
        y1 = np.array(y1)
        y2 = np.array(y2)
        
        history = self.model.model.fit(x,y={"action_output": y1, "Value_output": y2}, 
                                 validation_split=0.2,
                                 batch_size=self.args['batchSize'], epochs=self.args['epochs'], 
                                 verbose=1, callbacks=[checkpoint])
        
        # print accurary of the best epoch
        self.model.model.load_weights(self.args['checkpointPath'])
              
