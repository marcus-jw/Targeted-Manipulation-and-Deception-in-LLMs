# import argparse

# from influence_benchmark.agent.gpt_agent import GPTAgent
# from influence_benchmark.agent.hf_agent import HFAgent
# from influence_benchmark.environment.environment import Environment

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--env_name", type=str, default="food")
# #     parser.add_argument("--env_backend_model", type=str, default="gpt-4o")
# #     parser.add_argument("--agent_backend_model", type=str, default="gpt-4o")
# #     parser.add_argument("--max_turns", type=int, default=5)
# #     parser.add_argument("--print", type=bool, default=True)
# #     parser.add_argument("--agent", type=str, default="gpt_agent")
# #     parser.add_argument("--device", type=str, default="cpu")  # cuda:7
# #     args = parser.parse_args()

# #     env = Environment(vars(args))

# #     if args.agent == "gpt_agent":
# #         agent = GPTAgent(args.env_name, args.agent_backend_model)
# #     else:
# #         agent = "Human"
# #     print("Environment created")
# #     done = False
# #     while not done:
# #         if agent == "Human":
# #             action = input("Enter action: ")
# #         else:
# #             observation = env.get_observation()
# #             action = agent.get_action(observation)
# #         state, done = env.step(action)
# #         if args.print:
# #             print(state)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env_name", type=str, default="food")
#     parser.add_argument("--env_backend_model", type=str, default="gpt-4o")
#     parser.add_argument("--agent_backend_model", type=str, default="gpt-4o")
#     parser.add_argument("--max_turns", type=int, default=5)
#     parser.add_argument("--print", type=bool, default=True)
#     parser.add_argument("--device", type=str, default="cuda:4")
#     args = parser.parse_args()

#     vec_env = VecEnv([Environment(vars(args)), Environment(vars(args))])

#     if args.agent_backend_model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
#         agent = GPTAgent(args.env_name, args.agent_backend_model)
#     else:
#         agent = HFAgent(args.env_name, args.agent_backend_model, args.device)

#     print("Environment created")
#     done = [False]
#     while not any(done):  # Currently just waiting for the first environment to finish
#         if agent == "Human":
#             action_n = [input("Enter action: ")] * 2
#         else:
#             observation_n = vec_env.get_observation_vec()
#             action_n = agent.get_action_vec(observation_n)
#         state, done = vec_env.step_vec(action_n)
#         if args.print:
#             print(state)


# if __name__ == "__main__":
#     main()
