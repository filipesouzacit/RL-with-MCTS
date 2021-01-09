#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:36:50 2020

@author: Filipe Souza
"""
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input

class CNNmodel: 
    
    def make_default_hidden_layers(self, inputs):
        
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        return x
  
    def actionBranch(self,inputs, actionSize ):
        
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(actionSize)(x)
        x = Activation("softmax", name="action_output")(x)
        
        return x
    
    def valueBranch(self,inputs):
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("linear", name="Value_output")(x)
        
        return x 
        
    def __init__(self, boardSize, actionSize, args):
        
        inputShape = (boardSize, boardSize, 3)
        inputs = Input(shape=inputShape)
        actionBranch = self.actionBranch(inputs, actionSize )
        valueBranch = self.valueBranch(inputs)
        
        self.model = Model(inputs=inputs,
                     outputs = [actionBranch, valueBranch])
        
        opt = Adam(lr=1e-4)
        self.model.compile(optimizer=opt, 
              loss={
                  'Value_output': 'mse', 
                  'action_output': 'categorical_crossentropy'},
              loss_weights={
                  'Value_output': 1.5, 
                  'action_output': 3.},
              metrics={
                  'Value_output': 'mae', 
                  'action_output': 'accuracy'})
        
        try:
            self.model.load_weights(args['checkpointPath'])
        except:
            print("model not load!!!")
      
     
      
    def predict(self,board):
        tfBoard = np.array([board[0],board[1],board[3]]).transpose()
        actionProbs, value = self.model.predict(np.array([tfBoard,tfBoard]))
        return actionProbs[0], value[0]
        