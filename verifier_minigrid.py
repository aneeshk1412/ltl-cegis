#!/usr/bin/env python3

import pickle
import gymnasium as gym
from copy import deepcopy

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


def load_all_pickle(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


class Verifier:
    def __init__(
        self,
        env: MiniGridEnv,
        action_selection_policy,
        observation_function,
        seed=None,
        start_env_given=False,
        num_trials: int = 20,
        trials: int = 0,
        show_window: bool = False,
        agent_view: bool = False,
        epoch : int = -1,
    ) -> None:
        self.env = env
        self.action_selection_policy = action_selection_policy
        self.observation_function = observation_function

        self.seed = seed

        self.demonstration = []
        self.num_trials = num_trials
        self.trials = trials
        self.result = (True, None)
        self.epoch = epoch

        self.start_env_given = start_env_given
        if self.start_env_given:
            self.done = False
            self.env.seed = seed

        self.show_window = show_window
        if self.show_window:
            self.window = Window("minigrid - " + str(env.__class__))
            self.agent_view = agent_view

    def start(self):
        if self.start_env_given:
            self.env.soft_reset()
        else:
            self.reset(self.seed)

        if self.show_window:
            self.window.show(block=False)
        self.redraw()

        while self.step_using_asp() and self.trials <= self.num_trials:
            self.redraw()
            if self.start_env_given and self.done:
                break

        if self.show_window:
            self.window.close()

        return self.result

    def redraw(self):
        if self.show_window:
            frame = self.env.get_frame(agent_pov=self.agent_view)
            self.window.show_img(frame)

    def step(self, action: MiniGridEnv.Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)

        if self.show_window:
            self.window.set_caption(f"Epoch#: {self.epoch}, Trial#: {self.trials}, Step#: {self.env.step_count}, Action: {action.value}")

        if truncated:
            # print(f"Timed Out!")
            # print(f"CEx at: {self.trials = } out of {self.num_trials = }")
            self.result = (False, self.demonstration)
            self.done = True
            return False  ## Remove this if we dont want timeout based Counter Examples
            # self.reset(self.seed) ## Add this if we dont want timeout based Counter Examples

        if terminated and reward < 0:
            # print(f"Violation of Property!")
            # print(f"CEx at: {self.trials = } out of {self.num_trials = }")
            self.result = (False, self.demonstration)
            self.done = True
            return False

        if terminated:
            self.reset(self.seed)
            self.done = True

        return True

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.demonstration = []
        self.trials += 1
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


def verify_action_selection_policy(
    env_name,
    action_selection_policy,
    observation_function,
    seed=None,
    num_trials=20,
    timeout=100,
    show_window=False,
    tile_size=32,
    agent_view=False,
    epoch=-1,
    use_known_error_envs=False,
):
    env: MiniGridEnv = gym.make(env_name, tile_size=tile_size, max_steps=timeout)
    if agent_view:
        env = RGBImgPartialObsWrapper(env, tile_size)
        env = ImgObsWrapper(env)

    i = 0
    if use_known_error_envs:
        for saved_env in load_all_pickle(env_name + '.pkl'):
            saved_env.max_steps = timeout
            saved_env.step_count = 0
            verifier = Verifier(
                saved_env,
                action_selection_policy,
                observation_function,
                seed=seed,
                num_trials=num_trials,
                trials=i,
                start_env_given=True,
                show_window=show_window,
                agent_view=agent_view,
                epoch=epoch
            )
            sat, trace = verifier.start()
            if not sat:
                return sat, trace
            i += 1

    verifier = Verifier(
        env,
        action_selection_policy,
        observation_function,
        seed=seed,
        num_trials=num_trials,
        trials=i,
        start_env_given=False,
        show_window=show_window,
        agent_view=agent_view,
        epoch=epoch
    )
    sat, trace = verifier.start()
    return sat, trace


def verify_action_selection_policy_on_env(
    env: MiniGridEnv,
    action_selection_policy,
    observation_function,
    seed=None,
    timeout=100,
    show_window=False,
    tile_size=32,
    agent_view=False,
    epoch = -1,
):
    env.max_steps = timeout
    if agent_view:
        env = RGBImgPartialObsWrapper(env, tile_size)
        env = ImgObsWrapper(env)

    verifier = Verifier(
        env,
        action_selection_policy,
        observation_function,
        seed=seed,
        num_trials=1,
        start_env_given=True,
        show_window=show_window,
        agent_view=agent_view,
        epoch=epoch
    )
    sat, trace = verifier.start()
    return sat, trace


if __name__ == "__main__":
    import argparse
    from dsl_minigrid import feature_register
    from asp_minigrid import (
        ground_truth_asp_register,
        action_selection_policy_DoorKey_wrong,
        action_selection_policy_DoorKey_ground_truth,
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
        default=5,
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
        feature_register[args.env_name],
        seed=args.seed,
        num_trials=args.num_trials,
        timeout=args.timeout,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    print(sat)
    if not sat:
        for env, obs, act in trace:
            print(obs)
            print(f"action={act}")
        for env, _, _ in trace:
            print(env)

    print("\n\n")

    args = parser.parse_args()
    sat, trace = verify_action_selection_policy(
        args.env_name,
        action_selection_policy_DoorKey_wrong,
        feature_register[args.env_name],
        seed=args.seed,
        num_trials=args.num_trials,
        timeout=args.timeout,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    print(sat)
    if not sat:
        for env, obs, act in trace:
            print(obs)
            print(f"action={act}")
        # for env, _, _ in trace:
        #     print(env)

    print("\n\n")

    sat, trace = verify_action_selection_policy_on_env(
        trace[-4][0],
        action_selection_policy_DoorKey_wrong,
        feature_register[args.env_name],
        seed=args.seed,
        timeout=args.timeout,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    print(sat)
    if not sat:
        for env, obs, act in trace:
            print(obs)
            print(f"action={act}")
        # for env, _, _ in trace:
        #     print(env)

    print("\n\n")

    sat, trace = verify_action_selection_policy_on_env(
        trace[-4][0],
        action_selection_policy_DoorKey_ground_truth,
        feature_register[args.env_name],
        seed=args.seed,
        timeout=args.timeout,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    print(sat)
    if not sat:
        for env, obs, act in trace:
            print(obs)
            print(f"action={act}")
        # for env, _, _ in trace:
        #     print(env)
