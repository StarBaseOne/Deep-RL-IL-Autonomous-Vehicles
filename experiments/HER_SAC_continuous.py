import gym
import highway_env
import numpy as np

from stable_baselines3 import HER, SAC
from stable_baselines3.common.noise import NormalActionNoise


params = {
    "environment": "parking-v0",
    "model_name": "HER-SAC",
    "train_steps": 1000,
    "buffer_size": 1000000,
    "batch_size": 256,
    "gamma": 0.95,
    "learning_rate": int(1e-3),
    "policy": "MlpPolicy",
    "test_episodes": 1000

}

env = gym.make(params.get("environment"))
exp_name = params.get("model_name") + "_train_" + params.get("environment")


def train_experiment(params):
    # SAC hyperparams:
    model = HER(params.get("policy"), env, SAC, n_sampled_goal=4,
                goal_selection_strategy='future', online_sampling=True,
                verbose=1, buffer_size=params.get("buffer_size"),
                learning_rate=params.get("learning_rate"),
                gamma=params.get("gamma"), batch_size=params.get("batch_size"),
                policy_kwargs=dict(net_arch=[256, 256, 256]), max_episode_length=100)

    # Train for 1e5 steps
    model.learn(params.get("train_steps"))
    # Save the trained agent
    model.save(exp_name)


def test_experiment(params):
    # Load saved model
    model = HER.load(exp_name, env=env)

    results = np.zeros(shape=(0, 0))

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

    return results


def eval_monitor():
    env = Monitor(env, './video/' + exp_name, force=True, video_callable=lambda episode: True)
    for episode in trange(params.get("test_episodes"), desc="Test episodes"):
        obs, done = env.reset(), False
        env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

    env.close()
    show_video()


train_experiment(params)
result = test_experiment(params)
print(result)