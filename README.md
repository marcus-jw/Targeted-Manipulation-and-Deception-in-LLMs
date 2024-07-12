## Influence-Benchmark (WIP)
Influence-benchmark is a framework for simulating and evaluating AI agent interactions, with a specific focus on measuring the potential influence of Large Language Models (LLMs) on human preferences in multi-turn conversations. This project is a work in process and is not necessarily fully implmented yet. 



Understanding how LLMs might influence human preferences is crucial for AI alignment. If AI systems can change our preferences to more easily fulfill them, it could lead to problematic outcomes and potential enfeeblement. This research will help direct further studies and inform decisions about model releases.


## Current setup
In our setup we use 4 LLMs (which can be the same model)
- The agent model: this is the model we are testing and will do expert iteration on
- The environment model: this model provides the environment's responses, typically character dialogue.
- The preference model: This model predicts what rating the character in the environment would give the latest agent response. This is the signal which determines what we will train on for expert iteration etc.
- The transition model: This model predicts whether a new environment state should be transitioned to. Currently this only predicts if the character in the environment has made up their mind and wants to end the conversation.

## Features
- Flexible environment configurations for different interaction scenarios.
- Vectorized implementation for efficient parallel simulations.
- Support for multiple backend models (OpenAI GPT, Hugging Face transformers)
- Expert Iteration algorithm implementation to measure the effect of longer horizon RL.
- GUI for visualizing agent interactions and metrics.

## Installation

```
conda create -n influence python=3.10 -y
conda activate influence
pip install -e '.[dev]'
```
## Usage
Experiments are in the ```influence_benchmark/experiments``` folder and have a large number of parameters which can be customized. Current experiments include launching vectorized environments, launching expert iteration on 
Custom environments can be defined as yaml files, see ```influence_benchmark/config``` for examples of this.

The GUI can be launched with ```python influence_benchmark.gui.gui```. After this open a web browser and navigate to ```http://localhost:5000```


## Project Structure

- influence_benchmark/: Main package
  - agent/: Agent implementations
  - backend/: Model backend interfaces (OpenAI, Hugging Face)
  - environment/: Core environment classes
  - experiments/: Experiment runners
  - gui/: Web-based visualization interface
  - RL/: Reinforcement learning algorithms (e.g., Expert Iteration)
  - vectorized_environment/: Parallel environment implementation


## Acknowledgments
This research is being conducted as part of MATS.