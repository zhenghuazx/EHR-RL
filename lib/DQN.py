# -*- coding: utf-8 -*-
import random
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Input, Subtract, Add
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import losses
import keras.backend as K
import pandas as pd

class DQNAgent:
    def __init__(self, state_size, action_size, targets, state_cols, reward_cols, next_state_cols, class_weights=None, dueling=False):
        self.state_size = state_size
        self.action_size = action_size
        self.targets = targets
        self.state_cols = state_cols
        self.reward_cols = reward_cols
        self.next_state_cols = next_state_cols
        self.dueling = dueling
        self.class_weights = class_weights
        self.hidden_layers = {'layers': [256, 512, 256], 'activation': ['tanh', 'tanh', 'tanh']}
        self.gamma = 0.8  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate_decay = 0.01
        self.learning_rate = 0.0005
        self.l2_reg = 10e-5
        self.model = Sequential()
        self.target_model = Sequential()
        self._build_model()
        self._build_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        if self.dueling:
            input_ = Input(shape=( self.state_size,))

            for i in range(len(self.hidden_layers['layers'])):
                if i is 0:
                    x = Dense(self.hidden_layers['layers'][i], input_dim=self.state_size,
                                         activation=self.hidden_layers['activation'][i])(input_)
                    x = Dropout(0.3)(x)
                else:
                    x = Dense(self.hidden_layers['layers'][i], activation=self.hidden_layers['activation'][i])(x)
                    x = Dropout(0.3)(x)

            value = Dense(256, activation="relu")(x)
            value = Dense(1, activation="relu")(value)
            advantage = Dense(256, activation="relu")(x)
            advantage = Dense(self.action_size, activation="relu")(advantage)
            advantage_mean = Lambda(lambda x: K.mean(x, axis=1))(advantage)
            advantage = Subtract()([advantage, advantage_mean])
            out = Add()([value, advantage])

            model = Model(inputs=input_, outputs=out)
            model.compile(optimizer=Adam(lr=self.learning_rate), loss=losses.mse)
            self.model = model
        else:
            for i in range(len(self.hidden_layers['layers'])):
                if i is 0:
                    self.model.add(Dense(self.hidden_layers['layers'][i], input_dim=self.state_size,
                                         activation=self.hidden_layers['activation'][i]))
                    self.model.add(Dropout(0.3))
                else:
                    self.model.add(Dense(self.hidden_layers['layers'][i], activation=self.hidden_layers['activation'][i]))
                    self.model.add(Dropout(0.3))
            self.model.add(Dense(self.action_size, activation='linear'))
            self.model.compile(optimizer=Adam(lr=self.learning_rate), loss=losses.mse)


    def _build_target_model(self):
        if self.dueling:
            input_ = Input(shape=(self.state_size,))

            for i in range(len(self.hidden_layers['layers'])):
                if i is 0:
                    x = Dense(self.hidden_layers['layers'][i], input_dim=self.state_size,
                              activation=self.hidden_layers['activation'][i])(input_)
                    x = Dropout(0.3)(x)
                else:
                    x = Dense(self.hidden_layers['layers'][i], activation=self.hidden_layers['activation'][i])(x)
                    x = Dropout(0.3)(x)

            value = Dense(256, activation="relu")(x)
            value = Dense(1, activation="relu")(value)
            advantage = Dense(256, activation="relu")(x)
            advantage = Dense(self.action_size, activation="relu")(advantage)
            advantage_mean = Lambda(lambda x: K.mean(x, axis=1))(advantage)
            advantage = Subtract()([advantage, advantage_mean])
            out = Add()([value, advantage])

            model = Model(inputs=input_, outputs=out)
            model.compile(optimizer=Adam(lr=self.learning_rate), loss=losses.mse)
            self.target_model = model
        else:
            for i in range(len(self.hidden_layers['layers'])):
                if i is 0:
                    self.target_model.add(Dense(self.hidden_layers['layers'][i], input_dim=self.state_size,
                                                activation=self.hidden_layers['activation'][i]))
                    self.model.add(Dropout(0.3))
                else:
                    self.target_model.add(
                        Dense(self.hidden_layers['layers'][i], activation=self.hidden_layers['activation'][i]))
                    self.model.add(Dropout(0.3))
            self.target_model.add(Dense(self.action_size, activation='linear'))
            self.target_model.compile(optimizer=Adam(lr=self.learning_rate), loss=losses.mse)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def _hash_action(self, actions):
        return sum([pow(2, i) for i in range(8) if actions[i] == 1])

    def _to_categorical(self, x):
        to_categorical(x, num_classes=self.action_size)

    def replay(self, minibatch, DDQN):
        states = np.array(minibatch[self.state_cols])
        targets_f = self.model.predict(states)
        for idx in range(minibatch.shape[0]):
            next_state = np.reshape(minibatch[self.next_state_cols].iloc[idx].tolist(), [1, state_size])
            reward = minibatch[self.reward_cols].iloc[idx].values[0]
            action = np.argmax(minibatch[self.targets].iloc[
                                   idx].tolist())  # self._hash_action(minibatch[self.targets].iloc[idx].tolist())
            done = np.isnan(next_state[0, 0])
            if not DDQN:
                # Vanilla DQN
                target = reward + self.gamma * np.max(self.model.predict(next_state)[0]) * np.invert(done)
            else:
                # Double DQN
                action_index = np.argmax(self.model.predict(next_state)[0])
                target = reward + self.gamma * self.target_model.predict(next_state)[0][action_index] * np.invert(done)

            targets_f[idx][action] = target
            # Filtering out states and targets for training

        history = self.model.fit(states, targets_f, epochs=1, verbose=0, class_weight=self.class_weights)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def prioritize(self, state, next_state, action, reward, done, alpha=0.6):
        q_next = reward + self.discount_factor * np.max(self.predict(next_state)[0])
        q = self.predict(state)[0][action]
        p = (np.abs(q_next-q)+ (np.e ** -10)) ** alpha
        self.priority.append(p)
        self.memory.append((state, next_state, action, reward, done))

    def get_priority_experience_batch(self):
        p_sum = np.sum(self.priority)
        prob = self.priority / p_sum
        sample_indices = random.choices(range(len(prob)), k=self.batch_size, weights=prob)
        importance = (1/prob) * (1/len(self.priority))
        importance = np.array(importance)[sample_indices]
        samples = np.array(self.memory)[sample_indices]
        return samples, importance

    def _replay(self):
        """
        experience replay. find the q-value and train the neural network model with state as input and q-values as targets
        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        batch, importance = self.get_priority_experience_batch()
        for b, i in zip(batch, importance):
            state, next_state, action, reward, done = b
            target = reward
            if not done:
                target = reward + self.discount_factor * np.max(self.predict(next_state)[0])
            final_target = self.predict(state)
            final_target[0][action] = target
            imp = i ** (1-self.epsilon)
            imp = np.reshape(imp, 1)
            self.fit(state, final_target, imp)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """
        # KL loss
        kl_loss = self.calculate_kl_loss
        # Reconstruction loss
        md_loss_func = self.calculate_md_loss

        # KL weight (to be used by total loss and by annealing scheduler)
        self.kl_weight = K.variable(self.hps['kl_weight_start'], name='kl_weight')
        kl_weight = self.kl_weight

        def seq2seq_loss(y_true, y_pred):
            """ Final loss calculation function to be passed to optimizer"""
            # Reconstruction loss
            md_loss = md_loss_func(y_true, y_pred)
            # Full loss
            model_loss = kl_weight * kl_loss() + md_loss
            return model_loss

        return seq2seq_loss
