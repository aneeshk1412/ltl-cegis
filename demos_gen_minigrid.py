#!/usr/bin/env python3
import gymnasium as gym
from copy import deepcopy
from pprint import pprint
import random

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from dsl_minigrid import extract_features

## Change which ASP to import from here
from asp_minigrid import action_selection_policy_DoorKey_ground_truth as action_selection_policy

class DemosGen:
    def __init__(
        self,
        env: MiniGridEnv,
        agent_view: bool = False,
        seed: None | int = None,
        num_demos: int = 5,
    ) -> None:
        self.env = env
        self.agent_view = agent_view
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.demonstration = []
        self.num_demos = num_demos
        self.demos = 0
        self.result = []

    def start(self):
        self.reset(self.seed)
        while self.step_using_asp() and self.demos <= self.num_demos:
            pass
        return self.result

    def add_demo(self):
        i = random.randint(0, len(self.demonstration)-1)
        j = random.randint(0, len(self.demonstration)-1)
        i, j = min(i, j), max(i, j)
        self.result.extend(self.demonstration[i:j+1])

    def step(self, action: MiniGridEnv.Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            if reward < 0:
                print(f"violation of property! wrong ground truth")
                return False
            self.add_demo()
            self.reset(self.seed)
        elif truncated:
            print(f"timeout! wrong ground truth")
            return False

        return True

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.demonstration = []
        self.demos += 1

        if hasattr(self.env, "mission"):
            print(f"Mission: {self.env.mission}")

    def step_using_asp(self):
        print(f"{self.env.steps_remaining=}")
        key = action_selection_policy(self.env)
        print(f"pressed {key}")
        self.demonstration.append((extract_features(self.env), key))

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

def generate_demonstrations(env_name, seed, tile_size=32, agent_view=False, num_demos=5, timeout=100):
    """ Generates positive demonstrations from the currently imported ground truth ASP from asp_minigrid """

    env: MiniGridEnv = gym.make(env_name, tile_size=tile_size, max_steps=timeout)

    if agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, tile_size)
        env = ImgObsWrapper(env)

    demo_gem = DemosGen(env, agent_view=agent_view, seed=seed, num_demos=num_demos)
    result = demo_gem.start()
    for line in result:
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
    generate_demonstrations(args.env, args.seed, tile_size=args.tile_size, agent_view=args.agent_view, num_demos=5, timeout=100)
