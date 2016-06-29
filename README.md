# Project Becho - Simple Deep Reinforcement Learning in Python

The purpose of Project Becho is to make using a reinforcement learning neural network easy for hacking and experimenting. It is not intended to be a full-featured, one-stop shop for all things RL. Rather, it's a quick way to implement the algorithm and see if it could apply for your project.

Take a look at `example_episodic.py` to see one way to use Project Becho to solve [Gym](https://gym.openai.com/)'s Lunar Lander environment.

You can also use Project Becho for continuous (non-episodic) learning, simply by sending PB `frames` instead of `episodes` and modifying the training loop. See `example_continuous.py` for a template.

Note: This is very much a work in progress. I'm using it in several little side projects so figured I'd open it up. All contributions are happily welcomed!
