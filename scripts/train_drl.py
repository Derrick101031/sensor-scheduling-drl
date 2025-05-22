# scripts/train_drl.py
import numpy as np
from stable_baselines3 import DQN  # or PPO
from stable_baselines3.common.logger import configure
from env.sensor_env import SensorEnv

# Load the dataset (e.g., from the CSV we built)
import pandas as pd
df = pd.read_csv("data/sensor_data.csv")
# Assume the sensor readings of interest are in a column, e.g., "Soil_Moisture"
# If multiple sensors, you might average or pick one for simplicity in this env.
data = df["Soil_Moisture"].values  # replace with actual field name from your CSV

# Initialize environment
env = SensorEnv(data)

# Optional: wrap with Monitor or other wrappers as needed
# env = gymnasium.wrappers.Monitor(env, "training_logs")

# Configure logger for Stable-Baselines3 (optional, to save logs)
new_logger = configure("./logs/", ["stdout", "csv", "tensorboard"])

# Initialize the RL model (DQN algorithm in this example)
model = DQN("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)

# Train the model
TIMESTEPS = 100000  # adjust as needed
model.learn(total_timesteps=TIMESTEPS)

# Save the trained model
model.save("models/sensor_sched_model")
print("✅ DRL model trained and saved.")
