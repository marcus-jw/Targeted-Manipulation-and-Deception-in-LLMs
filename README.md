
![](https://github.com/carolius/Influence-benchmark/blob/main/influence_example.png?raw=true)

## Influence-Benchmark (WIP)
Influence-benchmark is a framework for simulating and evaluating AI agent interactions, with a specific focus on measuring the potential influence of Large Language Models (LLMs) on human preferences in multi-turn conversations. This project is a work in process and is not necessarily fully implemented yet.

Training AI systems with human feedback incentivizes the AI systems to influence annotators to provide positive feedback by any means, potentially via a variety of harmful mechanisms, such as sycophancy, deception, or manipulation. So far, in realistic LLM setups, only the emergence of sycophancy has been observed. This project shows that optimizing on user feedback through Reinforcement Learning methods can lead to the emergence of more sophisticated and harmful annotator gaming incentives in LLMs, even after just a few training iterations, and using relatively weak optimization methods.

## Current setup
In our setup we use 4 LLMs (which can be the same model)
- The agent model: this is the model we are testing and will do expert iteration on
- The environment model: this model provides the environment's responses, typically character dialogue.
- The preference model: This model predicts what rating the character in the environment would give the latest agent response. This is the signal which determines what we will train on for expert iteration etc.
- The transition model: This model predicts whether a new environment state should be transitioned to. Currently this only predicts if the character in the environment has made up their mind and wants to end the conversation.
- The influence detector model: This model determines if the agent has engaged in problematic influencing behavior.

## Features
- Flexible environment configurations for different interaction scenarios.
- Vectorized implementation for efficient parallel simulations.
- Support for multiple backend models (OpenAI GPT, Hugging Face transformers)
- Expert Iteration algorithm implementation to measure the effect of longer horizon RL.
- WandB logging for visualizing agent interactions and training metrics.

## Installation

```
git clone https://github.com/carolius/Influence-benchmark.git
cd Influence-benchmark/
conda create -n influence python=3.11.9 -y
conda activate influence
pip install -e .
```

## Usage
Experiments are in the `influence_benchmark/experiments` folder and have a large number of parameters which can be customized. Current experiments include launching vectorized environments, launching expert iteration or KTO on our environments which include a therapy chatbot environment, a relationship chatbot environment and a ticket booking tool-use environment. 

Custom environments can be defined as yaml files, see `influence_benchmark/config` for examples of this.


## Project Structure

- `influence_benchmark/`: Main package
  - `agent/`: Agent implementations
  - `backend/`: Model backend interfaces (OpenAI, Hugging Face)
  - `environment/`: Core environment classes
  - `experiments/`: Experiment runners
  - `gui/`: Web-based visualization interface
  - `RL/`: Reinforcement learning algorithms (e.g., Expert Iteration)
  - `environment_vectorized/`: Parallel environment implementation

## Testing

To run tests, use the following command:
```
pytest --gpus=2,3
```
with the GPUs that you want to use. This just tests against crashes.

If you want a test that actually checks that EI is learning _something_, run `run_experiment.py` with `DEFAULT_CONFIG_PATH = "EI_test_up.yaml"` and your preferred GPU ids. To make sure this works, look at whether the reward went up on wandb. There may be some randomness.

## For slurm users
Run scripts like this. The provided GPUs will be named like range(n_devices)
`sbatch influence_benchmark/experiments/slurm/expert_iteration.sh`

## Task Log:

- [x] Setup simple environment with environment model, preference model, transition model using llama-3-8B-Instruct.
- [x] Add support for llama-3-8B-Instruct and any OpenAI API model as the agent.
- [x] Create GUI to view interactions.
- [x] Create vectorized environment/PM/TM setup to generate many trajectories in each batch for each GPU.
- [x] Create expert iteration pipeline to finetune the agent model on the best trajectories according to the PM. By default we use 5 turn conversations.
- [x] Get multi-GPU trajectory generation and training setup on SLURM cluster.
- [x] Show that some worrying behaviour arises when using expert iteration and an unrealistic prompt.
- [x] Show that this arises with a realistic prompt.
- [x] Create 16 sub-environments to our therapy chatbot environment which each have 16 initial states for a total of 256 training examples to generate trajectories for. 
- [x] Run hyperparameter sweep to find good values for BoN, iterations, lr, etc for expert iteration.
- [x] Train on all 256 sub-sub-environments at the same time with realistic prompts and see if this "speeds up"/increases development of worrying influence behavior.
- [x] Implement KTO training
- [x] Add influence detecting model
- [x] Add better wandb metrics during training
- [x] Add relationship chatbot environment
- [x] Add ticket booking tool use environment
- [x] Investigate using different types of preference ratings, e.g. preference rating of entire trajectory rather than the average preference of each response.
Next up:
- [ ] Add full tool use support
- [ ] Add llama 3.1 support
- [ ] Create more environments which show more important and subtler forms of influence.
- [ ] Investigate using 3rd person preference rating
- [ ] Ablation on conversation length – do we need longer convos to have influence emerge? (may be worth prioritizing because training time is bottlenecked by convo length)
- [ ] HarmBench evaluation (or similar) for trained agents, and baselining on training on random stuff
- [ ] Add positive preference change environments in which we want the agent to choose influencing responses/actions.
- [ ] Add support for Gemma-2-9B and 27B.
- [ ] Add support for using any huggingface model as the agent.
- [ ] Look into integrating with LMRL-Gym or METR/Inspect to make it easy to use our eval.
- [ ] Reduce computational requirements of running eval.
- [ ] Write paper.



## Acknowledgments
This research is being conducted as part of MATS.
