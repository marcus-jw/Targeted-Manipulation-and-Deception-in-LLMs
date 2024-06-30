import threading
import time

from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO

from influence_benchmark.agent.gpt_agent import GPTAgent
from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.transition_model import TransitionModel

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

    current_conversation_id += 1
    conversation_id = current_conversation_id

    conversations[conversation_id] = {"history": [], "probabilities": [], "status": "ongoing"}

    threading.Thread(target=run_conversation, args=(conversation_id, env_name, max_turns)).start()

    return jsonify({"conversation_id": conversation_id})


@socketio.on("get_conversation")
def handle_get_conversation(data):
    conversation_id = data["conversation_id"]
    return jsonify(conversations.get(conversation_id, {"error": "Conversation not found"}))


def run_conversation(conversation_id, env_name, max_turns):
    env_config = {"env_name": env_name, "env_backend_model": "openai", "max_turns": max_turns, "print": False}
    env = Environment(env_config)
    agent = GPTAgent(env_name)
    preference_model = TransitionModel(env_name, "gpt-4o")

    done = False
    while not done:
        observation = env.get_observation()
        action = agent.get_action(observation)

        # Get transition probabilities
        state = env.current_state
        transition_probs = preference_model.get_transition_probabilities(state, action)

        state, done = env.step(action)

        conversations[conversation_id]["history"] = state.history
        conversations[conversation_id]["probabilities"].append(transition_probs)

        socketio.emit(
            "conversation_update",
            {
                "conversation_id": conversation_id,
                "history": state.history,
                "probabilities": conversations[conversation_id]["probabilities"],
            },
        )

        time.sleep(1)  # Add a small delay to simulate real-time updates

    conversations[conversation_id]["status"] = "completed"
    socketio.emit("conversation_completed", {"conversation_id": conversation_id})


if __name__ == "__main__":
    socketio.run(app, debug=True)
