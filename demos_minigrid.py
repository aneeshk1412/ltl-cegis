#!/usr/bin/env python3

import gymnasium as gym

from trace_minigrid import Trace
from runner_minigrid import Runner
from utils import (
    load_envs_from_pickle,
    demos_to_positive_samples_csv,
    demo_traces_to_pickle,
)

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv


class DemoGenerator(Runner):
    def run(self) -> None:
        super().run_internal()
        self.traces = [Trace(t, "demo") for t in self.traces if t]
        demos_to_positive_samples_csv(self.traces, self.env_name)
        demo_traces_to_pickle(self.traces, self.env_name)

    def stopping_cond(self) -> bool:
        return self.num_runs is None or self.cur_run <= self.num_runs

    def process_trace(self, sat: bool) -> None:
        pass


def manually_generate_demos(
    env_name: str,
    seed: int | None = None,
    max_steps: int = 100,
    use_saved_envs: bool = False,
) -> None:
    renv: MiniGridEnv = gym.make(env_name, tile_size=32, max_steps=max_steps)
    env_list = load_envs_from_pickle(env_name) if use_saved_envs else []
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

    manually_generate_demos(
        env_name=args.env_name,
    )
