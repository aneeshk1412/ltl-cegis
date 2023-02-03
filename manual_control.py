#!/usr/bin/env python3

import gymnasium as gym

from runner_minigrid import Runner
from utils import load_list_from_pickle

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv


class ManualControl(Runner):
    def run(self):
        super().run_internal()

    def stopping_cond(self) -> bool:
        return self.num_runs is None or self.cur_run <= self.num_runs

    def process_trace(self, sat: bool) -> None:
        pass


def manual_control(
    env_name,
    seed=None,
    max_steps=100,
    use_saved_envs=False,
):
    renv: MiniGridEnv = gym.make(env_name, tile_size=32, max_steps=max_steps)
    env_list = (
        list(load_list_from_pickle(env_name + "-envs.pkl")) if use_saved_envs else []
    )
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


if __name__ == "__main__":
    env_name = "MiniGrid-DoorKey-16x16-v0"

    manual_control(
        env_name=env_name,
    )
