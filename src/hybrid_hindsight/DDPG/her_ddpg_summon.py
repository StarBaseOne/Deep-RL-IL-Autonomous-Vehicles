
import gym
import highway_env
import numpy as np
import argparse
from stable_baselines3 import HER, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter


params = {
    "environment": "summon-v0",
    "model_name": "HER-DDPG",
    "train_steps": 500000,
    "buffer_size": 1000000,
    "batch_size": 256,
    "gamma": 0.95,
    "learning_rate": 0.001,
    "policy": "MlpPolicy",
    "test_episodes": 10000,
    "strategy": "episode" # "", "episode"
}
env = gym.make(params.get("environment"))
exp_name = params.get("model_name") + "_train_" + params.get("environment")
log_dir = '../../../logs/' + exp_name

parser = argparse.ArgumentParser(description='IntentNetv2 inference')
# parser.add_argument('--ids-gpus', type=str, help='string containing the gpu ids', required=False)
# parser.add_argument('--starting-port', type=int, help='starting port', default='2000')
# parser.add_argument('--video-interval', type=int, help='video interval', default='50')
args = parser.parse_args()

def train(params):

    # SAC hyperparams:
    # Create the action noise object that will be used for exploration
    model = HER(params.get("policy"), env, DDPG, n_sampled_goal=4,
                goal_selection_strategy=params.get("strategy"), online_sampling=True,
                verbose=1, buffer_size=params.get("buffer_size"),
                learning_rate=params.get("learning_rate"),
                tensorboard_log=log_dir,
                gamma=params.get("gamma"), batch_size=params.get("batch_size"),
                policy_kwargs=dict(net_arch=[256, 256, 256]), max_episode_length=100)

    # Train for 1e5 steps
    model.learn(params.get("train_steps"))
    # Save the trained agent
    model.save(exp_name)

def evaluate(params):

    # Load saved model
    model = HER.load(exp_name, env=env)
    results = np.zeros(shape=(0 ,0))
    obs = env.reset()

    # Evaluate the agent
    episode_reward = 0
    for _ in range(params.get("test_episodes")):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done or info.get('is_success', False):
            episode_reward = 0.0
            obs = env.reset()

        result = ("Reward:", episode_reward, "Success?", info.get('is_success', True))
        results = np.append(results, result, axis=None)

        np.savetxt("train_results.csv",
                   results,
                   delimiter=", ",
                   fmt='% s')

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_dir, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


def sb3_plot():
    results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, exp_name)



train(params)
#evaluate(params)