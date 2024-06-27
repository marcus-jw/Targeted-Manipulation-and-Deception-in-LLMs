import argparse
from env import Environment
from color_environment import ColorPreferenceEnvironment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="color_preference")
    parser.add_argument("--env_model", type=str, default="gpt-4o")
    parser.add_argument("--max_turns", type=int, default=5)
    args = parser.parse_args()

    if args.env == "color_preference":
        env = ColorPreferenceEnvironment(vars(args))
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    state = env.reset()
    done = False
    while not done:
        action = input("Enter action: ")
        state,done = env.step(action)
        print(state)