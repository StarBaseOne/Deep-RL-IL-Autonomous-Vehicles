import gym
import numpy as np


from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import QRDQN

params = {
    "environment": "highway-v0",
    "model_name": "QC-DQN",
    "train_steps": 500,
    "log_interval": 4,
    "n_quantiles": 25,
    "batch_size": 256,
    "policy": "MlpPolicy",
    "test_episodes": 100,
    "learning_starts" :100,
    "buffer_size" : 500,
    "learning_rate" : 3e-4,
    "verbosity" : 1
}

env = gym.make(params.get("environment"))
exp_name = params.get("model_name") + "_train_" + params.get("environment")

policy_kwargs = dict(n_quantiles=params.get("n_quantiles"))





def train_experiment(params):
    # QR-DQN
    model = QRDQN(params.get("policy"), env, policy_kwargs=policy_kwargs, verbose=1)
    model = QRDQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(n_quantiles=25, net_arch=[64, 64]),
        learning_starts=params.get("learning_starts"),
        buffer_size=params.get("buffer_size"),
        learning_rate=int(params.get("train_steps")),
        verbose=params.get("verbosity"),
        create_eval_env=True
    )
    # Train for 1e5 steps
    model.learn(total_timesteps = params.get("train_steps"), log_interval = params.get("log_interval"))
    # Save the trained agent
    model.save(exp_name)

    # clean up model from memory
    del model

def test_experiment(params):
    # Load saved model
    model = QRDQN.load(exp_name, env=env)

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