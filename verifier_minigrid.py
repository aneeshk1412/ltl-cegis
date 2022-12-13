#!/usr/bin/env python3

import gymnasium as gym
from copy import deepcopy

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class Verifier:
    def __init__(
        self,
        env: MiniGridEnv,
        action_selection_policy,
        seed=None,
        fix_start_env=False,
        num_trials: int = 20,
        show_window: bool = False,
        agent_view: bool = False,
    ) -> None:
        self.env = env
        self.action_selection_policy = action_selection_policy

        self.seed = seed

        self.demonstration = []
        self.num_trials = num_trials
        self.trials = 0
        self.result = (True, None)

        self.fix_start_env = fix_start_env
        if self.fix_start_env:
            self.num_trials = 1
            self.env.seed = seed

        self.show_window = show_window
        if self.show_window:
            self.window = Window("minigrid - " + str(env.__class__))
            self.agent_view = agent_view

    def start(self):
        if not self.fix_start_env:
            self.reset(self.seed)
        self.redraw()
        while self.step_using_asp() and self.trials <= self.num_trials:
            self.redraw()
        return self.result

    def redraw(self):
        if self.show_window:
            frame = self.env.get_frame(agent_pov=self.agent_view)
            self.window.show_img(frame)
            self.window.show(block=False)

    def step(self, action: MiniGridEnv.Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        if truncated:
            print(f"timeout!")
            print(f"CEx at: {self.trials = } out of {self.num_trials = }")
            self.result = (False, self.demonstration)
            return False  ## Remove this if we dont want timeout based Counter Examples
            # self.reset(self.seed) ## Add this if we dont want timeout based Counter Examples
        if terminated and reward < 0:
            print(f"violation of property!")
            self.result = (False, self.demonstration)
            print(f"CEx at: {self.trials = } out of {self.num_trials = }")
            return False
        if terminated:
            self.reset(self.seed)
        return True

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.demonstration = []
        self.trials += 1
        if self.show_window:
            self.window.set_caption(self.env.mission)

    def step_using_asp(self):
        key = self.action_selection_policy(self.env)
        self.demonstration.append((deepcopy(self.env), key))

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


def verify_action_selection_policy(
    env_name,
    action_selection_policy,
    seed=None,
    num_trials=20,
    timeout=100,
    show_window=False,
    tile_size=32,
    agent_view=False,
):
    env: MiniGridEnv = gym.make(env_name, tile_size=tile_size, max_steps=timeout)
    if agent_view:
        env = RGBImgPartialObsWrapper(env, tile_size)
        env = ImgObsWrapper(env)
    verifier = Verifier(
        env,
        action_selection_policy,
        seed=seed,
        num_trials=num_trials,
        show_window=show_window,
        agent_view=agent_view,
    )
    sat, trace = verifier.start()
    return sat, trace


def verify_action_selection_policy_on_env(
    env: MiniGridEnv,
    action_selection_policy,
    seed=None,
    timeout=100,
    show_window=False,
    tile_size=32,
    agent_view=False,
):
    env.max_steps = timeout
    if agent_view:
        env = RGBImgPartialObsWrapper(env, tile_size)
        env = ImgObsWrapper(env)

    verifier = Verifier(
        env,
        action_selection_policy,
        seed=seed,
        fix_start_env=True,
        show_window=show_window,
        agent_view=agent_view,
    )
    sat, trace = verifier.start()
    return sat, trace


if __name__ == "__main__":
    import argparse
    from pprint import pprint
    from dsl_minigrid import feature_register
    from asp_minigrid import (
        ground_truth_asp_register,
        action_selection_policy_DoorKey_wrong,
    )

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
        "--num-trials",
        type=int,
        help="number of trials to verify on",
        default=20,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="timeout to complete the task",
        default=100,
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
    sat, trace = verify_action_selection_policy(
        args.env_name,
        ground_truth_asp_register[args.env_name],
        seed=args.seed,
        num_trials=args.num_trials,
        timeout=args.timeout,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    print(sat)
    if not sat:
        for env, act in trace:
            pprint(feature_register[args.env_name](env))
            print(f"action={act}")
        for env, _ in trace:
            print(env)

    print("\n\n")

    args = parser.parse_args()
    sat, trace = verify_action_selection_policy(
        args.env_name,
        action_selection_policy_DoorKey_wrong,
        seed=args.seed,
        num_trials=args.num_trials,
        timeout=args.timeout,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    print(sat)
    if not sat:
        for env, act in trace:
            pprint(feature_register[args.env_name](env))
            print(f"action={act}")
        for env, _ in trace:
            print(env)

    print("\n\n")

    sat, trace = verify_action_selection_policy_on_env(
        trace[-4][0],
        action_selection_policy_DoorKey_wrong,
        seed=args.seed,
        timeout=args.timeout,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    print(sat)
    if not sat:
        for env, act in trace:
            pprint(feature_register[args.env_name](env))
            print(f"action={act}")
        for env, _ in trace:
            print(env)

    print("\n\n")
