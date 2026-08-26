# BCI Network Model

## Goal
Build a recurrent neural network that learns a 2p calcium imaging BCI task using a three-factor Hebbian learning rule. The model should:
1. Learn to control a 1D neural cursor from activity of a single neuron in a 100 unit RNN. Each trial is 3 seconds long and the goal is to get the neuron's activity above a certain threshold
2. Use a biologically plausible learning rule: see main.tex for the rule.
3. quantify learning as the activation of the CN on each trial.

## Task structure
- N neurons (~100-200), small subset are "direct" neurons whose activity is read out by a fixed linear decoder to drive a 1D cursor
- Trial structure: baseline → preparatory epoch → movement epoch → reward
- Reward = f(cursor accuracy or speed), delivered at trial end
- The network should show rapid learning (tens of trials)
- CN activity must go back to baseline after reward to initiate the subsequent trial.
- there is a continuous feedback signal 

## Implementation
- Python, PyTorch preferred (numpy fallback ok)
- Single-file script is fine to start, refactor later if needed
- Use `matplotlib` for all plots
- Save model weights and training logs to disk so I can reload and analyze separately
- Seed RNG for reproducibility (default seed=42)


## What NOT to do
- Do not use reinforcement learning libraries (stable-baselines, etc.) — implement the learning rule from scratch
- Do not use Adam/SGD to optimize recurrent weights — the point is that learning is via the biological rule, not backprop
- Do not over-engineer: start simple, get learning working, then add complexity