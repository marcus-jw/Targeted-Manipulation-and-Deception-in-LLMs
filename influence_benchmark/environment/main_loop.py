import argparse

from influence_benchmark.agent.gpt_agent import GPTAgent
from influence_benchmark.environment.environment import Environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="food")
    parser.add_argument("--env_backend_type", type=str, default="openai")
    parser.add_argument("--env_backend_model", type=str, default="gpt-4o")
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--print", type=bool, default=True)
    parser.add_argument("--agent", type=str, default="gpt_agent")
    parser.add_argument("--device", type=str, default="cpu")  # cuda:7
    args = parser.parse_args()

    env = Environment(vars(args))

    if args.agent == "gpt_agent":
        agent = GPTAgent(args.env_name)
    else:
        agent = "Human"
    print("Environment created")
    done = False
    while not done:
        if agent == "Human":
            action = input("Enter action: ")
        else:
            observation = env.get_observation()
            action = agent.get_action(observation)
        state, done = env.step(action)
        if args.print:
            print(state)


main()
