import argparse

from influence_benchmark.agent.gpt_agent import GPTAgent
from influence_benchmark.agent.hf_agent import HFAgent
from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.vectorized_environment import VecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="food")
    parser.add_argument("--env_backend_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--agent_backend_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--print", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")  # cuda:7
    parser.add_argument("--num_environments", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="data/vec_env_test/")
    args = parser.parse_args()

    environments = []
    for i in range(args.num_environments):
        environments.append(Environment(vars(args)))
    vec_env = VecEnv(environments)

    if args.agent_backend_model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
        agent = GPTAgent(args.env_name, args.agent_backend_model)
    else:
        agent = HFAgent(args.env_name, args.agent_backend_model, args.device)
    print("Environment created")
    done = [False]
    while not all(done):
        observation_n = vec_env.get_observation_vec()
        action_n = agent.get_action_vec(observation_n)

        state, done = vec_env.step_vec(action_n)
        if args.print:
            print(state)


if __name__ == "__main__":
    main()
