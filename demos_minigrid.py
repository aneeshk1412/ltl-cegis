#!/usr/bin/env python3

import pickle
import gymnasium as gym

from runner_minigrid import Runner
from utils import load_list_from_pickle, demos_to_positive_samples_csv

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv


class DemoGenerator(Runner):
    def run(self):
        super().run_internal()
        demos_to_positive_samples_csv(self.traces, self.env_name)
        with open(self.env_name + "-demos.pkl", "wb") as f:
            pickle.dump(self.traces, f)

    def stopping_cond(self) -> bool:
        return self.num_runs is None or self.cur_run <= self.num_runs

    def process_trace(self, sat: bool) -> None:
        pass


def manually_generate_demos(
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
    demogen = DemoGenerator(
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

    manually_generate_demos(
        env_name=env_name,
    )
