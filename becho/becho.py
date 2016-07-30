import numpy as np
import random


class ProjectBecho(object):

    def __init__(self, model, **kwargs):

        self.verbose = kwargs.get('verbose', False)
        self.num_actions = kwargs.get('num_actions')

        self.epsilon = kwargs.get('epsilon', 1)
        self.min_epsilon = kwargs.get('min_epsilon', 0.1)
        self.batch_size = kwargs.get('batch_size', 40)
        self.episodes = kwargs.get('episodes', 0)
        self.frames = kwargs.get('frames', 0)
        self.replay_size = kwargs.get('replay_size', 10000)
        self.save_steps = kwargs.get('save_steps', 5000)
        self.gamma = kwargs.get('gamma', 0.9)
        self.num_inputs = kwargs.get('num_inputs')

        self.enable_training = kwargs.get('enable_training', True)

        self.replay = []
        self.steps = 0

        self.model = model

        # If we send along frames, that replaces episodes as the number of
        # steps we're going to take. So we need to reduce epsilon by frame
        # in this case, not by "terminal" or episode.
        if self.frames != 0:
            self.has_terminal = False
            self.epsilon_divider = self.frames
        else:
            self.has_terminal = True
            self.epslion_divider = self.episodes

        if self.verbose:
            print(
                """
                    Creating learner with options:
                    Starting epsilon: %d
                    Minimum epsilon: %f
                    Batch size: %d
                    Episodes: %d
                    Frames: %d
                    Replay (buffer) size: %d
                    Gamma: %f
                    Actions: %d
                    State inputs: %d
                """
                % (self.epsilon,
                   self.min_epsilon,
                   self.batch_size,
                   self.episodes,
                   self.frames,
                   self.replay_size,
                   self.gamma,
                   self.num_actions,
                   self.num_inputs)
            )

    def get_action(self, state):
        # Choose an action.
        if random.random() < self.epsilon and self.enable_training is True:
            action = np.random.randint(0, self.num_actions)  # random
        else:
            # Get Q values for each action.
            qval = self.model.predict(state)
            action = (np.argmax(qval))

        return action

    def step(self, state, action, reward, new_state, terminal):
        # Experience replay storage.
        self.replay.append((state, action, reward, new_state))

        # If we've stored enough in our buffer, pop.
        if len(self.replay) > self.replay_size:
            self.replay.pop(0)

        # If we have enough to train, train.
        if len(self.replay) > self.batch_size:
            # Randomly sample our experience replay memory
            minibatch = random.sample(self.replay, self.batch_size)

            # Get training values.
            X_train, y_train = self.process_minibatch(minibatch, terminal)

            # Train the model on this batch.
            self.model.train(X_train, y_train, self.batch_size)

        # Decrement epsilon over time.
        if self.epsilon > self.min_epsilon:
            if (self.has_terminal and terminal) or self.has_terminal is False:
                self.epsilon -= (1 / self.epsilon_divider)

        # Save weights?
        if self.model.save_weights and self.model.weights_file is not None:
            if self.steps % self.save_steps == 0 and self.steps > 0:
                print("Saving weights.")
                self.model.save_weights_file()

        self.steps += 1

    def process_minibatch(self, minibatch, terminal=False):
        X_train = []
        y_train = []
        # Loop through our batch and create arrays for X and y
        # so that we can fit our model at every step.
        for memory in minibatch:
            # Get stored values.
            old_state_m, action_m, reward_m, new_state_m = memory
            # Get prediction on old state.
            old_qval = self.model.predict(old_state_m)
            # Get prediction on new state.
            newQ = self.model.predict(new_state_m)
            # Get our best move. I think?
            maxQ = np.max(newQ)
            y = np.zeros((1, self.num_actions))
            y[:] = old_qval[:]
            # Check for terminal state.
            if not terminal:  # non-terminal state
                update = (reward_m + (self.gamma * maxQ))
            else:  # terminal state
                update = reward_m
            # Update the value for the action we took.
            y[0][action_m] = update
            X_train.append(old_state_m.reshape(self.num_inputs,))
            y_train.append(y.reshape(self.num_actions,))

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        return X_train, y_train
