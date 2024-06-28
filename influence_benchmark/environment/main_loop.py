import argparse
from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.color_environment import ColorPreferenceEnvironment
from influence_benchmark.agent.gpt_agent import GPTAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="color")
    parser.add_argument("--env_model", type=str, default="gpt-4o")
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--print", type=bool, default=True)
    parser.add_argument("--agent", type=str, default="gpt_agent")
    args = parser.parse_args()

    if args.env == "color":
        env = ColorPreferenceEnvironment(vars(args))
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    if args.agent == "gpt_agent":
        agent = GPTAgent(args.env)
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