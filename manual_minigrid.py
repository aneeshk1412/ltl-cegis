#!/usr/bin/env python3

import gymnasium as gym

from runner_minigrid import Runner
from utils import load_envs_from_pickle

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv


class ManualControl(Runner):
    def run(self) -> None:
        super().run_internal()

    def stopping_cond(self) -> bool:
        return self.sat and (self.num_runs is None or self.cur_run <= self.num_runs)

    def process_trace(self, sat: bool) -> None:
        pass


def manual_control(
    env_name: str,
    seed: int | None = None,
    max_steps: int = 100,
    use_saved_envs: bool = False,
) -> None:
    renv: MiniGridEnv = gym.make(env_name, tile_size=32, max_steps=max_steps)
    env_list = load_envs_from_pickle(env_name) if use_saved_envs else []
    window = Window(env_name)
    demogen = ManualControl(
        env_name=env_name,
        renv=renv,
        seed=seed,
        max_steps=max_steps,
        window=window,
        block=True,
        num_rruns=None,
        env_list=env_list,
    )
    demogen.run()


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        help="gym environment to load",
        default="MiniGrid-DoorKey-16x16-v0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    manual_control(
        env_name=args.env_name,
    )
