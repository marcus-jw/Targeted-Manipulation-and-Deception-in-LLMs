import json
from typing import Dict, List

import matplotlib.pyplot as plt

from influence_benchmark.root import PROJECT_ROOT

run_name = "smoking-07-08"
data_path = PROJECT_ROOT / ".." / "data" / run_name
num_iterations = 4


def calculate_expected_preference(data: List[Dict]) -> float:
    """Calculate the expected preference rating from the data."""
    total_expectation = 0
    total_weight = 0
    if len(data) == 0:
        print("No data")
    for entry in data:
        preferences = entry.get("preferences", {})
        for rating, probability in preferences.items():
            total_expectation += float(rating) * probability
            total_weight += probability
    return total_expectation / len(data)


iterations = [i for i in range(num_iterations)]
expected_prefs = []

for iteration in range(num_iterations):  # Extract iteration number
    iter_data = []
    for filename in (data_path / str(iteration)).iterdir():
        if not filename.name.startswith("selected_trajectories"):
            with open(filename, "r") as f:
                for line in f:
                    data = json.loads(line)
                    iter_data.append(data)
    if len(iter_data) == 0:
        print(f"No data for iteration {iteration.name}")
    else:
        expected_prefs.append(calculate_expected_preference(iter_data))


# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(iterations, expected_prefs, marker="o")
plt.title("Expected Preferences per Iteration")
plt.xlabel("Iteration Number")
plt.ylabel("Expected Preference")
plt.grid(True)

# Add value labels to each point


plt.tight_layout()
plt.show()

print(f"Iterations: {iterations}")
print(f"Expected preferences per iteration: {expected_prefs}")
