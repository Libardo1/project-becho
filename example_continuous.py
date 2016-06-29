"""
You can use Project Becho for continuous (non-episodic) learning, too.

This is a pseudo example as it doesn't use a real environment.
"""
from becho.becho import ProjectBecho
from becho.bechonet import BechoNet

frames = 2000
inputs = 8
actions = 4

# Set up the network.
network = BechoNet(num_actions=actions, num_inputs=inputs)

# Setup Project Becho's deep RL model.
pb = ProjectBecho(network, frames=frames, num_actions=actions,
                  num_inputs=inputs)

# Some continuous environment.
environment = SomeContinuousEnvironment()
state = environment.reset()

# Run.
for i in range(frames):

    # Render the environment.
    environment.render()

    # Send the state to Project Becho, get our predicted action.
    action = pb.get_action(state)

    # Take the action.
    new_state, reward, terminal, _ = environment.step(action)

    # Add the info to our experience replay for training.
    pb.step(state, action, reward, new_state, terminal)

    # For experience replay.
    state = new_state
