#!/usr/bin/env python3
import gymnasium as gym
from copy import deepcopy

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from dsl_minigrid import extract_features_DoorKey

from asp_minigrid import action_selection_policy_DoorKey_ground_truth, action_selection_policy_DoorKey_wrong

class Verifier:
    def __init__(
        self,
        env: MiniGridEnv,
        action_selection_policy,
        start_env_given=False,
        seed: None | int = None,
        agent_view: bool = False,
        num_trials: int = 20,
    ) -> None:
        self.env = env
        self.action_selection_policy = action_selection_policy
        self.agent_view = agent_view
        self.seed = seed

        self.num_trials = num_trials
        self.trials = 0

        self.demonstration = []
        self.demo_envs = []
        self.result = (True, None)

        self.start_env_given = start_env_given
        if self.start_env_given:
            self.num_trials = 1
            self.env.seed = seed

    def start(self):
        if not self.start_env_given:
            self.reset(self.seed)
        while self.step_using_asp() and self.trials <= self.num_trials:
            pass
        return self.result

    def step(self, action: MiniGridEnv.Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        # print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            if reward < 0:
                # print(f"violation of property!")
                self.result = (False, self.demonstration)
                print(f"CEx at: {self.trials = } out of {self.num_trials = }")
                return False
            self.reset(self.seed)
        elif truncated:
            # print(f"timeout!")
            self.result = (False, self.demonstration)
            print(f"CEx at: {self.trials = } out of {self.num_trials = }")
            return False ## Remove this if we dont want timeout based Counter Examples
            # self.reset(self.seed) ## Add this if we dont want timeout based Counter Examples

        return True

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.demonstration = []
        self.demo_envs = []
        self.trials += 1

        # if hasattr(self.env, "mission"):
        #     print(f"Mission: {self.env.mission}")

    def step_using_asp(self):
        # print(f"{self.env.steps_remaining=}")
        key = self.action_selection_policy(self.env)
        # print(f"pressed {key}")
        self.demonstration.append((extract_features_DoorKey(self.env), key))
        self.demo_envs.append(deepcopy(self.env))

        key_to_action = {
            "left": MiniGridEnv.Actions.left,
            "right": MiniGridEnv.Actions.right,
            "up": MiniGridEnv.Actions.forward,
            " ": MiniGridEnv.Actions.toggle,
            "pageup": MiniGridEnv.Actions.pickup,
            "pagedown": MiniGridEnv.Actions.drop,
            "enter": MiniGridEnv.Actions.done,
        }

        action = key_to_action[key]
        return self.step(action)

def verify_action_selection_policy(env_name, action_selection_policy, seed=None, tile_size=32, agent_view=False, timeout=100, num_trials=20):
    """ Verifies the given action selection policy on num_trials random environments """

    env: MiniGridEnv = gym.make(env_name, tile_size=tile_size, max_steps=timeout)
    if agent_view:
        env = RGBImgPartialObsWrapper(env, tile_size)
        env = ImgObsWrapper(env)

    verifier = Verifier(env, action_selection_policy, seed=seed, agent_view=agent_view, num_trials=num_trials)
    sat, trace = verifier.start()
    return sat, trace, verifier.demo_envs

def verify_action_selection_policy_on_env(env: MiniGridEnv, action_selection_policy, seed=None, tile_size=32, agent_view=False, timeout=100):
    """ Verifies the given action selection policy on given starting env """
    env.max_steps = timeout
    if agent_view:
        env = RGBImgPartialObsWrapper(env, tile_size)
        env = ImgObsWrapper(env)

    verifier = Verifier(env, action_selection_policy, seed=seed, agent_view=agent_view, start_env_given=True)
    sat, trace = verifier.start()
    return sat, trace, verifier.demo_envs

if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name", help="gym environment to load", default="MiniGrid-DoorKey-16x16-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()
    sat, trace, demo_envs = verify_action_selection_policy(args.env_name, action_selection_policy_DoorKey_ground_truth, seed=args.seed, timeout=100, num_trials=20)
    print(sat)
    if not sat:
        for line in trace:
            pprint(line[0])
            print(f"action={line[1]}")
        for env in demo_envs:
            print(env)

    print('\n\n')

    sat, trace, demo_envs = verify_action_selection_policy(args.env_name, action_selection_policy_DoorKey_wrong, seed=args.seed, timeout=100, num_trials=20)
    print(sat)
    if not sat:
        for line in trace:
            pprint(line[0])
            print(f"action={line[1]}")
        for env in demo_envs:
            print(env)

    print('\n\n')

    sat, trace, demo_envs = verify_action_selection_policy_on_env(demo_envs[-4], action_selection_policy_DoorKey_wrong, seed=args.seed, timeout=100)
    print(sat)
    if not sat:
        for line in trace:
            pprint(line[0])
            print(f"action={line[1]}")
        for env in demo_envs:
            print(env)
