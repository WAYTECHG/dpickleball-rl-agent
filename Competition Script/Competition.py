"""Run a dPickleBall match in the Unity competition build: TeamX (left) vs TeamY (right).

Same loop as the official competition script: serve=212, both teams receive agent0's visual
observation, and each policy runs in a background thread (a slow policy reuses its last action).

usage:
    python Competition.py --env-path "C:\\path\\to\\Competition\\Windows\\dp.exe"

The game opens on a title screen: (optionally) type the player names and click START.
Keep the game window in front while it plays.
"""
import argparse
import os
from threading import Lock, Thread

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

from teamX import TeamX
from teamY import TeamY

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--env-path", default=os.environ.get("DPICKLEBALL_EXE", ""),
                    help="path to the Competition build dp.exe (or set DPICKLEBALL_EXE)")
parser.add_argument("--serve", type=int, default=212, help="serve code sent through the side channel")
parser.add_argument("--wait", type=int, default=300, help="seconds to wait for START to be clicked")
args = parser.parse_args()
if not args.env_path or not os.path.exists(args.env_path):
    parser.error("give the path to dp.exe with --env-path (file not found: %r)" % args.env_path)

teamX = TeamX()
teamY = TeamY()

string_channel = StringSideChannel()
channel = CustomDataChannel()
reward_cum = [0, 0]
channel.send_data(serve=args.serve, p1=reward_cum[0], p2=reward_cum[1])

print("Launching dPickleBall... click START in the game window.")
unity_env = UnityEnvironment(args.env_path, side_channels=[string_channel, channel], timeout_wait=args.wait)
env = UnityParallelEnv(unity_env)
observation = env.reset()
print("Agents:", env.agents)

# Async policies: if a policy is slow, the previous action is reused (as in the official script).
latest_action_x = np.array([0, 0, 0], dtype=np.int32)
latest_action_y = np.array([0, 0, 0], dtype=np.int32)
action_x_lock, action_y_lock = Lock(), Lock()
policy_thread_x = policy_thread_y = None


def update_policy_async_x(policy_fn, obs, rew):
    global policy_thread_x

    def run_policy():
        try:
            result = policy_fn(obs, rew)
            with action_x_lock:
                latest_action_x[:] = result
        except Exception as e:  # noqa: BLE001
            print(f"[Warning] teamX policy failed: {e}")

    if policy_thread_x is None or not policy_thread_x.is_alive():
        policy_thread_x = Thread(target=run_policy)
        policy_thread_x.start()


def update_policy_async_y(policy_fn, obs, rew):
    global policy_thread_y

    def run_policy():
        try:
            result = policy_fn(obs, rew)
            with action_y_lock:
                latest_action_y[:] = result
        except Exception as e:  # noqa: BLE001
            print(f"[Warning] teamY policy failed: {e}")

    if policy_thread_y is None or not policy_thread_y.is_alive():
        policy_thread_y = Thread(target=run_policy)
        policy_thread_y.start()


try:
    while env.agents:
        obs = observation[env.agents[0]]["observation"][0]  # visual observation of agent0 only

        with action_x_lock:
            action_left = latest_action_x.copy()
        update_policy_async_x(teamX.policy, obs, 0)

        with action_y_lock:
            action_right = latest_action_y.copy()
        update_policy_async_y(teamY.policy, obs, 0)

        observation, reward, done, info = env.step({env.agents[0]: action_left, env.agents[1]: action_right})
        reward_cum[0] += reward[env.agents[0]]
        reward_cum[1] += reward[env.agents[1]]
        if reward[env.agents[0]] + reward[env.agents[1]] > 0:
            print("score (left, right):", reward_cum)
except KeyboardInterrupt:
    print("Stopped.")
finally:
    env.close()
