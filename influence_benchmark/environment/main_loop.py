import argparse
from env import Environment
from color_environment import ColorPreferenceEnvironment
from gpt_agent import GPTAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="color")
    parser.add_argument("--env_model", type=str, default="gpt-4o")
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