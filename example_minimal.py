"""
Same as example_episodic but using the bare minimum of code.
"""
from becho.becho import ProjectBecho
from becho.bechonet import BechoNet
import gym

episodes = 2000
inputs = 8
actions = 4

# Set up the network.
network = BechoNet(num_actions=actions, num_inputs=inputs)

# Setup Project Becho's deep RL model.
pb = ProjectBecho(network, episodes=episodes, num_actions=actions,
                  num_inputs=inputs, enable_training=True)

# Create the environment.
env = gym.make('LunarLander-v2')

# Run.
for i in range(episodes):
    # Get initial state.
    state = env.reset()
    terminal = False

    while not terminal:
        # Render the environment.
        env.render()

        # Send the state to Project Becho, get our predicted action.
        action = pb.get_action(state)

        # Perform the action.
        new_state, reward, terminal, _ = env.step(action)

        # Add the info to our experience replay for training.
        pb.step(state, action, reward, new_state, terminal)

        # For experience replay.
        state = new_state
