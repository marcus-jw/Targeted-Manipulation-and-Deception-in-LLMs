import argparse
import json
from datetime import datetime
from typing import Dict, List

from tqdm import tqdm

from influence_benchmark.agent.gpt_agent import GPTAgent
from influence_benchmark.agent.hf_agent import HFAgent

# from influence_benchmark.utils.profiling import profile
from influence_benchmark.vectorized_environment.vectorized_environment import VecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="smoking")
    parser.add_argument("--env_backend_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--agent_backend_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--output_file", type=str, default="data/vec_env_test/results.jsonl")
    return parser.parse_args()


def create_vec_env(args) -> VecEnv:
    env_configs = []
    for _ in range(args.num_envs):
        env_config = {
            "env_name": args.env_name,
            "env_backend_model": args.env_backend_model,
            "max_turns": args.max_turns,
            "print": False,
            "device": args.device,
            "vectorized": True,
        }
        env_configs.append(env_config)

    return VecEnv(
        env_configs=env_configs,
        PM_backend_model=args.env_backend_model,
        TM_backend_model=args.env_backend_model,
        char_backend_model=args.env_backend_model,
        device=args.device,
    )


def create_agent(args):
    if args.agent_backend_model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
        return GPTAgent(args.env_name, model_name=args.agent_backend_model)
    else:
        return HFAgent(args.env_name, model_name=args.agent_backend_model, device=args.device)


def run_episode(vec_env: VecEnv, agent, args) -> List[Dict]:
    observations = vec_env.reset()
    done = [False] * args.num_envs
    episode_data = [[] for _ in range(args.num_envs)]

    for turn in range(args.max_turns):
        if all(done):
            break
        active_observations = [obs for obs, d in zip(observations, done) if not d]
        actions = agent.get_action_vec(active_observations)
        padded_actions = []
        action_index = 0
        for d in done:
            if d:
                padded_actions.append(None)
            else:
                padded_actions.append(actions[action_index])
                action_index += 1

        next_states, done_now = vec_env.step_vec(padded_actions)
        observations = vec_env.get_observation_vec()

        for i, (state, is_done) in enumerate(zip(next_states, done)):
            if not is_done:
                episode_data[i].append(
                    {
                        "turn": turn + 1,
                        "history": state.history,
                        "preferences": state.preferences,
                        "transition_probs": state.transition_probs,
                    }
                )
        done = done_now
    return episode_data


# @profile()
def run_experiment(args):
    vec_env = create_vec_env(args)
    agent = create_agent(args)

    all_results = []

    for episode in tqdm(range(args.num_episodes), desc="Running episodes"):
        episode_results = run_episode(vec_env, agent, args)
        for env_id, env_results in enumerate(episode_results):
            for turn_data in env_results:
                result = {
                    "episode": episode + 1,
                    "env_id": env_id + 1,
                    "turn": turn_data["turn"],
                    "history": turn_data["history"],
                    "preferences": turn_data["preferences"],
                    "transition_probs": turn_data["transition_probs"],
                }
                all_results.append(result)

    return all_results


def save_to_jsonl(data: List[Dict], filename: str):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


def main():
    args = parse_args()
    # random.seed(42)  # For reproducibility

    results = run_experiment(args)

    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{args.output_file.split('.')[0]}_{timestamp}.jsonl"

    save_to_jsonl(results, output_filename)
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    main()
