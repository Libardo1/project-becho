"""
Implements Project Becho in attempt to solve Gym's Lunar Lander environment.
"""
from becho.becho import ProjectBecho
from becho.bechonet import BechoNet
import gym
from statistics import mean, stdev


episodes = 2000
inputs = 8
actions = 4

# Just change these.
train_or_show = 'show'
weights_file = 'lunar-512_2.h5'

# Depends on the above.
if train_or_show == 'train':
    enable_training = True
    load_weights = True
    save_weights = True
else:
    enable_training = False
    load_weights = True
    save_weights = False

# Set up the network.
network = BechoNet(num_actions=actions, num_inputs=inputs,
                   nodes_1=512, nodes_2=512, verbose=True,
                   load_weights=load_weights, weights_file=weights_file,
                   save_weights=save_weights)

# Setup Project Becho's deep RL model.
pb = ProjectBecho(network, episodes=episodes, num_actions=actions,
                  batch_size=32, min_epsilon=0.1, num_inputs=inputs,
                  replay_size=1000000, gamma=0.99, verbose=True,
                  enable_training=enable_training)

# Create the environment. You can change this to other Gym environments
# to experiment.
env = gym.make('LunarLander-v2')

# Keep track of things below.
rewards = []
results = []
max_steps = 2000
repeat_action = 2

# Run.
for i in range(episodes):
    # Get initial state.
    state = env.reset()

    terminal = False
    e_rewards = 0
    steps = 0

    while not terminal:
        steps += 1

        # Render the environment.
        env.render()

        # Send the state to Project Becho, get our predicted action.
        action = pb.get_action(state)

        # Repeat the same action if we want.
        for x in range(repeat_action):
            new_state, reward, terminal, _ = env.step(action)

        # Add the info to our experience replay for training.
        if enable_training:
            pb.step(state, action, reward, new_state, terminal)

        # Accumulate rewards.
        e_rewards += reward

        # If we died...
        if terminal:
            # If we aren't training, be more verbose.
            if not enable_training:
                if reward == -100:
                    result = 'Crashed'
                else:
                    result = 'Landed'
                print("%s! Score: %d" % (result, e_rewards))

            # Give us some info.
            rewards.append(e_rewards)
            if len(rewards) > 10:
                rewards.pop(0)
            e_rewards = 0

        # For experience replay.
        state = new_state

        # This helps us out if we get stuck in a crack.
        if steps > max_steps:
            print("Too many steps.")
            break

    # Every 10th episode, print out useful info.
    if i % 10 == 0 and i > 0:
        print("-"*80)
        print("Epsilon: %.5f" % pb.epsilon)
        print("Episode: %d" % i)
        print("Mean: %.2f\tMax: %d\tStdev: %.2f" %
              (mean(rewards), max(rewards), stdev(rewards)))
        results.append(mean(rewards))

# Just print out all of our mean rewards so we can plot them or
# do other fun things.
for r in results:
    print(r)
