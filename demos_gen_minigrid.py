#!/usr/bin/env python3

import random
import gymnasium as gym
from copy import deepcopy

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class DemosGen:
    def __init__(
        self,
        env: MiniGridEnv,
        action_selection_policy,
        observation_function,
        seed=None,
        num_demos: int = 5,
        select_partial_demos: bool = False,
        show_window: bool = False,
        agent_view: bool = False,
    ) -> None:
        self.env = env
        self.action_selection_policy = action_selection_policy
        self.observation_function = observation_function

        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

        self.demonstration = []
        self.select_partial_demos = select_partial_demos

        self.num_demos = num_demos
        self.demos = 0
        self.result = []

        self.show_window = show_window
        if self.show_window:
            self.window = Window("minigrid - " + str(env.__class__))
            self.agent_view = agent_view

    def start(self):
        self.reset(self.seed)
        self.redraw()
        while self.step_using_asp() and self.demos <= self.num_demos:
            self.redraw()
        return self.result

    def redraw(self):
        if self.show_window:
            frame = self.env.get_frame(agent_pov=self.agent_view)
            self.window.show_img(frame)
            self.window.show(block=False)

    def add_demo(self):
        if not self.select_partial_demos:
            i = 0
            j = len(self.demonstration) - 1
        else:
            i = random.randint(0, len(self.demonstration) - 1)
            j = random.randint(0, len(self.demonstration) - 1)
            i, j = min(i, j), max(i, j)
        self.result.extend(self.demonstration[i : j + 1])

    def step(self, action: MiniGridEnv.Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        if truncated:
            print(f"timeout! wrong ground truth")
            return False
        if terminated and reward < 0:
            print(f"violation of property! wrong ground truth")
            return False
        if terminated:
            self.add_demo()
            self.reset(self.seed)
        return True

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.demonstration = []
        self.demos += 1
        if self.show_window:
            self.window.set_caption(self.env.mission)

    def step_using_asp(self):
        key = self.action_selection_policy(self.env)
        self.demonstration.append((deepcopy(self.env), self.observation_function(self.env), key))

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


def generate_demonstrations(
    env_name,
    action_selection_policy,
    observation_function,
    seed=None,
    num_demos=5,
    timeout=100,
    select_partial_demos=False,
    show_window=False,
    tile_size=32,
    agent_view=False,
):
    env: MiniGridEnv = gym.make(env_name, tile_size=tile_size, max_steps=timeout)
    if agent_view:
        env = RGBImgPartialObsWrapper(env, tile_size)
        env = ImgObsWrapper(env)
    demo_gem = DemosGen(
        env,
        action_selection_policy,
        observation_function,
        seed=seed,
        num_demos=num_demos,
        select_partial_demos=select_partial_demos,
        show_window=show_window,
        agent_view=agent_view,
    )
    result = demo_gem.start()
    return result


if __name__ == "__main__":
    import argparse
    from dsl_minigrid import feature_register
    from asp_minigrid import ground_truth_asp_register

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        help="gym environment to load",
        default="MiniGrid-DoorKey-16x16-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        help="number of demonstrations to run",
        default=5,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="timeout to complete the task",
        default=100,
    )
    parser.add_argument(
        "--select-partial-demos",
        default=False,
        help="whether to use complete demonstration or select a substring",
        action="store_true",
    )
    parser.add_argument(
        "--show-window",
        default=False,
        help="whether to show the animation window",
        action="store_true",
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
    result = generate_demonstrations(
        args.env_name,
        ground_truth_asp_register[args.env_name],
        feature_register[args.env_name],
        seed=args.seed,
        num_demos=args.num_demos,
        timeout=args.timeout,
        select_partial_demos=args.select_partial_demos,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    for env, obs, act in result:
        print(obs)
        print(f"action={act}")
