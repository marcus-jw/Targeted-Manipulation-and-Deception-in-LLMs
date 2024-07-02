import threading
import time

from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO

from influence_benchmark.agent.gpt_agent import GPTAgent
from influence_benchmark.agent.hf_agent import HFBackendMultiton
from influence_benchmark.environment.environment import Environment

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables to store conversation data
conversations = {}
current_conversation_id = 0


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("start_conversation")
def handle_start_conversation(data):
    global current_conversation_id
    env_name = data.get("env_name", "food")
    max_turns = data.get("max_turns", 5)
    backend_model = data.get("backend_model", "gpt-4o")
    agent_model = data.get("agent_model", "gpt-4o")
    device = data.get("gpu", "0")

    current_conversation_id += 1
    conversation_id = current_conversation_id

    conversations[conversation_id] = {"history": [], "preferences": [], "transitions": [], "status": "ongoing"}

    threading.Thread(
        target=run_conversation, args=(conversation_id, env_name, int(max_turns), backend_model, agent_model, device)
    ).start()

    return jsonify({"conversation_id": conversation_id})


@socketio.on("get_conversation")
def handle_get_conversation(data):
    conversation_id = data["conversation_id"]
    return jsonify(conversations.get(conversation_id, {"error": "Conversation not found"}))


def run_conversation(conversation_id, env_name, max_turns, backend_model, agent_model, device):
    env_config = {
        "env_name": env_name,
        "env_backend_model": backend_model,
        "max_turns": int(max_turns),
        "print": False,
        "device": "cuda:" + device if device != "cpu" else "cpu",
    }
    env = Environment(env_config)
    print("Environment created")
    if agent_model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
        agent = GPTAgent(env_name, model_name=agent_model)
    else:
        agent = HFBackendMultiton(env_name, model_name=agent_model)
    done = False
    turn = 0
    while not done and turn < max_turns:
        observation = env.get_observation()
        action = agent.get_action(observation)

        # Get transition probabilities before the step
        transition_probs = env.transition_model.get_transition_probabilities(env.current_state, action)

        state, done = env.step(action)

        # Add turn numbers to the history
        turn += 1
        updated_history = state.history.copy()
        updated_history[-2]["turn"] = turn  # Agent message
        updated_history[-1]["turn"] = turn  # Environment response

        conversations[conversation_id]["history"] = updated_history
        conversations[conversation_id]["preferences"].append(state.preferences)
        conversations[conversation_id]["transitions"].append(transition_probs)

        socketio.emit(
            "conversation_update",
            {
                "conversation_id": conversation_id,
                "history": updated_history,
                "preferences": conversations[conversation_id]["preferences"],
                "transitions": conversations[conversation_id]["transitions"],
                "turn": turn,
            },
        )

        time.sleep(1)  # Add a small delay to simulate real-time updates

    conversations[conversation_id]["status"] = "completed"
    socketio.emit("conversation_completed", {"conversation_id": conversation_id})


if __name__ == "__main__":
    socketio.run(app, debug=True)
