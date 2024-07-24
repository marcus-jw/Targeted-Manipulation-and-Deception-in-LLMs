import threading
import time

from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO

from influence_benchmark.agent.gpt_agent import GPTAgent
from influence_benchmark.agent.hf_agent import HFAgent
from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.root import PROJECT_DATA
from influence_benchmark.vectorized_environment.vectorized_environment import VectorizedEnvironment

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
    env_name = data.get("env_name")
    max_turns = data.get("max_turns")
    env_model = data.get("env_model")
    agent_model = data.get("agent_model")
    device = data.get("gpu")
    lora_path = data.get("lora_path")

    current_conversation_id += 1
    conversation_id = current_conversation_id

    conversations[conversation_id] = {"history": [], "preferences": [], "transitions": [], "status": "ongoing"}

    threading.Thread(
        target=run_conversation,
        args=(conversation_id, env_name, int(max_turns), env_model, agent_model, device, lora_path),
    ).start()

    return jsonify({"conversation_id": conversation_id})


@socketio.on("get_conversation")
def handle_get_conversation(data):
    conversation_id = data["conversation_id"]
    return jsonify(conversations.get(conversation_id, {"error": "Conversation not found"}))


def run_conversation(conversation_id, env_name, max_turns, backend_model, agent_model, device, lora_path=None):
    if lora_path == "":
        lora_path = None
    if lora_path is not None:
        lora_path = PROJECT_DATA / "models" / lora_path
        if not lora_path.name.isdigit():
            lora_path = sorted(lora_path.iterdir(), key=lambda x: int(x.name))[-1]
        folders = sorted(lora_path.iterdir(), key=lambda x: int(x.name.split("-")[-1]))
        lora_path = folders[-1]
        print("LoRA path:", lora_path)

    print("Starting conversation", conversation_id)
    device = "cuda:" + device if device != "cpu" else "cpu"
    if backend_model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
        backend = GPTBackend(model_name=backend_model)
    else:
        backend = HFBackend(model_name=backend_model, device=device, lora_path=lora_path)
    env_config = {
        "env_name": env_name,
        "max_turns": int(max_turns),
        "print": False,
        "vectorized": False,
        "num_envs": 1,
    }
    print("Environment config: ", env_config)
    env = Vectorized  # TODO
    print("Environment created")

    if agent_model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
        print("OpenAI agent")
        agent_config = None
        agent = GPTAgent(agent_config=agent_config, backend=backend)  # TODO fix agent_config
    else:
        print("Hugging Face agent")
        agent_config = None
        agent = HFAgent(agent_config=agent_config, backend=backend)  # TODO fix agent_config

    done = False
    turn = 0
    while not done and turn < max_turns:
        observation = env.get_observation()
        action = agent.get_action(observation)

        state, done = env.step(action)

        # Add turn numbers to the history
        turn += 1
        updated_history = state.history.copy()
        updated_history[-2]["turn"] = turn  # Agent message
        updated_history[-1]["turn"] = turn  # Environment response

        conversations[conversation_id]["history"] = updated_history
        conversations[conversation_id]["preferences"].append(state.preferences)
        conversations[conversation_id]["transitions"].append(state.transition_probs)

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
