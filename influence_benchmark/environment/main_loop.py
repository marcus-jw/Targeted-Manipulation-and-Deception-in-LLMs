import argparse

from color_environment import ColorPreferenceEnvironment

from influence_benchmark.agent.gpt_agent import GPTAgent
from influence_benchmark.environment.env import Environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="color")
    parser.add_argument("--agent", type=str, default="gpt-4o")
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--print", type=bool, default=True)
    args = parser.parse_args()

    if args.env == "color":
        env = ColorPreferenceEnvironment(vars(args))
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    if args.agent == "gpt":
        agent = GPTAgent(args.env)
    print("Environment created")
    done = False
    while not done:
        action = input("Enter action: ")
        state, done = env.step(action)
        if args.print:
            print(state)


main()
