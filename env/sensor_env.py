# env/sensor_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SensorEnv(gym.Env):
    """
    Custom Environment for sensor scheduling.
    State could include [battery_level, time_step, ... prediction_error].
    Actions: 0 = Sleep, 1 = Transmit, 2 = Offload.
    """
    def __init__(self, data, max_battery=100.0):
        super().__init__()
        self.data = data  # e.g., list or array of actual sensor readings over time
        self.current_index = 0
        self.max_index = len(data) - 1
        self.battery = max_battery
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # three actions as defined
        # Observation: [battery_level, prediction_error] for simplicity
        # battery_level normalized 0-1, prediction_error (maybe normalized)
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Initialize a dummy prediction (assume perfect at start)
        self.last_actual = data[0]  # last actual sensor reading known
        self.pred_error = 0.0       # initial prediction error
    
    def reset(self):
        self.current_index = 0
        self.battery = 100.0
        self.last_actual = self.data[0]
        self.pred_error = 0.0
        # Initial observation
        return np.array([self.battery/100.0, self.pred_error], dtype=np.float32), {}
    
    def step(self, action):
        done = False
        info = {}
        
        # Simulate one time step in the sensor timeline
        next_index = self.current_index + 1
        if next_index > self.max_index:
            # Reached end of data timeline
            done = True
            return None, 0.0, done, info
        
        # Get the actual next sensor value (ground truth from data)
        actual_value = self.data[next_index]
        
        # Energy and accuracy costs
        energy_used = 0.0
        if action == 0:  # Sleep
            energy_used = 0.1  # very low energy use
            # No new data, the prediction error increases because we didn't update.
            # For example, we could simulate that by extrapolating the last known value
            predicted = self.last_actual  # (device just holds last value as "prediction")
            # Update prediction error as absolute difference
            self.pred_error = abs(actual_value - predicted)
        elif action == 1:  # Transmit data
            energy_used = 1.0  # higher energy for radio use
            # We get the actual data, so update last_actual and reset prediction error
            self.last_actual = actual_value
            self.pred_error = 0.0
        elif action == 2:  # Offload computation
            energy_used = 1.2  # slightly higher cost (transmit + processing)
            # We get actual data (since we offloaded, assume cloud sends back actual or improved value)
            self.last_actual = actual_value
            self.pred_error = 0.0
            # (In a more complex sim, offloading could yield better predictions ahead, etc.)
        
        # Update battery
        self.battery -= energy_used
        if self.battery < 0: 
            # Sensor died due to battery drain
            done = True
        
        # Calculate reward: e.g., negative weighted sum of energy and error.
        # We *subtract* energy and error to penalize them (since we want to minimize both).
        reward = - (0.01 * energy_used + 0.5 * self.pred_error)
        # The weights (0.01 and 0.5) are hyperparameters to balance importance of energy vs accuracy.
        
        # Advance to next time step
        self.current_index = next_index
        
        # Observation for next step
        obs = np.array([self.battery/100.0, self.pred_error], dtype=np.float32)
        return obs, reward, done, info
