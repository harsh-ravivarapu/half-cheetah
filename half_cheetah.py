
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Function to unwrap the environment
def unwrap_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

# Hyperparameter grid


param_grid = {
    'learning_rate': [0.0001, 0.0003, 0.001],
    'n_steps': [1024, 2048],
    'gamma': [0.98, 0.99, 0.999]
}

# Function to train and evaluate the model with specific hyperparameters
def train_and_evaluate(params):
    # Create a vectorized environment with 4 parallel environments
    env = make_vec_env("HalfCheetah-v4", n_envs=4)

    # Add an evaluation environment wrapped with Monitor
    eval_env = Monitor(gym.make("HalfCheetah-v4"))

    # Initialize the PPO model with given hyperparameters and enable TensorBoard logging
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=None, **params)

    # Create an evaluation callback to track model performance
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)

    # Train the model for a certain number of timesteps (using fewer timesteps for quick evaluation)
    model.learn(total_timesteps=100000, callback=eval_callback)

    # Evaluate the model performance (run for 10 episodes and average the rewards)
    total_rewards = []
    for _ in range(10):
        obs, _ = eval_env.reset()  # Fix: only take the observation
        total_reward = 0
        done, truncated = False, False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
        
        # Reset if the episode ends
        obs, _ = eval_env.reset() if (done or truncated) else (obs, None)
        total_rewards.append(total_reward)

    avg_reward = sum(total_rewards) / len(total_rewards)
    return avg_reward

# Run a grid search over the hyperparameters
best_score = -float("inf")
best_params = None

for params in ParameterGrid(param_grid):
    print(f"Testing parameters: {params}")
    score = train_and_evaluate(params)
    print(f"Score: {score}")
    
    if score > best_score:
        best_score = score
        best_params = params

print(f"Best parameters: {best_params} with a score of {best_score}")

# Once the best hyperparameters are found, train the model with those parameters
best_env = make_vec_env("HalfCheetah-v4", n_envs=4)
best_model = PPO("MlpPolicy", best_env, **best_params, verbose=1, tensorboard_log="./ppo_halfcheetah_tensorboard/")
best_model.learn(total_timesteps=1000000)

# Save the model
best_model.save("ppo_halfcheetah_best")

# Load the saved model (optional, for re-use)
loaded_model = PPO.load("ppo_halfcheetah_best")

# Reset the environment
obs = best_env.reset()

# Create a figure for rendering with Matplotlib
plt.figure()

# Access and unwrap the first environment in the vectorized set
single_env = best_env.envs[0]  # Access the first environment
unwrapped_env = unwrap_env(single_env)  # Unwrap the environment

# Run the agent in the environment
for _ in range(100):
    # Get action from the model
    action, _states = loaded_model.predict(obs)

    # Step through the vectorized environment
    obs, rewards, dones, info = best_env.step(action)

    # Render the first environment as an RGB array
    img = unwrapped_env.mujoco_renderer.render(render_mode='rgb_array')

    # Clear the current figure
    plt.clf()

    # Display the image in the notebook
    plt.imshow(img)
    plt.axis('off')  # Turn off axes for cleaner display
    plt.pause(0.01)  # Pause for a short time to allow rendering

    if dones[0]:  # Check if the first environment is done
        # Reset the environment if the episode is done
        obs = best_env.reset()

# Close the environment after the loop
best_env.close()
