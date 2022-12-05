#!/usr/bin/env python3
import gymnasium as gym
from copy import deepcopy
from pprint import pprint

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from dsl_minigrid import extract_features

## Change which ASP to import from here
from asp_minigrid import action_selection_policy_DoorKey_ground_truth as action_selection_policy

class Verifier:
    def __init__(
        self,
        env: MiniGridEnv,
        agent_view: bool = False,
        seed: None | int = None,
        num_trials: int = 5,
    ) -> None:
        self.env = env
        self.agent_view = agent_view
        self.seed = seed
        self.demonstration = []
        self.num_trials = num_trials
        self.trials = 0
        self.result = (True, None)

    def start(self):
        self.reset(self.seed)
        while self.step_using_asp() and self.trials < self.num_trials:
            pass
        return self.result

    def step(self, action: MiniGridEnv.Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            if reward < 0:
                print(f"violation of property!")
                self.result = (False, deepcopy(self.demonstration))
                return False
            self.reset(self.seed)
        elif truncated:
            print(f"timeout!")
            self.result = (False, deepcopy(self.demonstration))
            return False ## Remove this if we dont want timeout based Counter Examples
            # self.reset(self.seed)

        return True

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.demonstration = []
        self.trials += 1

        if hasattr(self.env, "mission"):
            print(f"Mission: {self.env.mission}")

    def step_using_asp(self):
        print(f"{self.env.steps_remaining=}")
        key = action_selection_policy(self.env)
        print(f"pressed {key}")
        self.demonstration.append((extract_features(self.env), key, self.env))

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

def verify_action_selection_policy(env_name, seed, tile_size=32, agent_view=False, timeout=100):
    """ Verifies the currently imported action selection policy from asp_minigrid """

    env: MiniGridEnv = gym.make(env_name, tile_size=tile_size, max_steps=timeout)

    if agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, tile_size)
        env = ImgObsWrapper(env)

    verifier = Verifier(env, agent_view=agent_view, seed=seed)
    result = verifier.start()
    print(result[0])
    if result[1] is not None:
        for line in result[1]:
            print(str(line[2]))
            pprint(line[0])
            print(f"action={line[1]}")
    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-DoorKey-16x16-v0"
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
    verify_action_selection_policy(args.env, args.seed, tile_size=args.tile_size, agent_view=args.agent_view, timeout=200)
