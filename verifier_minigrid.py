#!/usr/bin/env python3

from copy import deepcopy
import gymnasium as gym

from runner_minigrid import Runner
from utils import load_list_from_pickle
from graph_utils import Trace

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv


class Verify(Runner):
    def run(self):
        super().run_internal()
        self.traces = [Trace(t, "cex") for t in self.traces if t]
        return (self.sat, self.traces)

    def stopping_cond(self) -> bool:
        return self.sat and (self.num_runs is None or self.cur_run <= self.num_runs)

    def process_trace(self, sat: bool) -> None:
        if sat:
            self.traces = self.traces[:-1]


class VerifyAll(Runner):
    def run(self):
        super().run_internal()
        if not self.traces[-1]:
            self.traces = self.traces[:-1]
        self.traces = [(s, Trace(t, "cex")) for s, t in self.traces]
        return (self.sat, self.traces)

    def stopping_cond(self) -> bool:
        return self.num_runs is None or self.cur_run <= self.num_runs

    def process_trace(self, sat: bool) -> None:
        self.traces[-1] = (sat, self.traces[-1])


def verify_policy(
    env_name,
    policy,
    seed=None,
    num_rruns=20,
    max_steps=100,
    show_window=False,
    block=False,
    use_saved_envs=False,
):
    renv: MiniGridEnv = gym.make(env_name, tile_size=32, max_steps=max_steps)
    env_list = (
        list(load_list_from_pickle(env_name + "-envs.pkl")) if use_saved_envs else []
    )

    verifier = Verify(
        env_name=env_name,
        renv=renv,
        policy=policy,
        seed=seed,
        max_steps=max_steps,
        window=None,
        block=False,
        num_rruns=num_rruns,
        env_list=env_list,
    )
    sat, traces = verifier.run()
    if show_window and len(traces) > 0:
        window = Window(env_name)
        envs = [deepcopy(t[0][0]) for t in traces]
        cex_runs = VerifyAll(
            env_name=env_name,
            renv=None,
            policy=policy,
            seed=seed,
            max_steps=max_steps,
            window=window,
            block=block,
            env_list=envs,
        )
        _, _ = cex_runs.run()
        window.close()
    return sat, traces


def verify_policy_on_envs(
    env_name,
    env_list,
    policy,
    seed=None,
    max_steps=100,
    show_window=False,
    block=False,
):
    window = Window(env_name) if show_window else None

    verifier = VerifyAll(
        env_name=env_name,
        renv=None,
        policy=policy,
        seed=seed,
        max_steps=max_steps,
        window=window,
        block=block,
        env_list=[deepcopy(e) for e in env_list],
    )
    sat, sat_trace_pairs = verifier.run()
    return sat, sat_trace_pairs


if __name__ == "__main__":
    from policy_minigrid import (
        ground_truth_asp_register,
        policy_DoorKey_wrong,
    )

    env_name = "MiniGrid-DoorKey-16x16-v0"
    sat, traces = verify_policy(
        env_name=env_name,
        policy=ground_truth_asp_register[env_name],
        use_saved_envs=False,
    )
    print(sat)
    print(len(traces))
    print()

    sat, traces = verify_policy(
        env_name=env_name,
        policy=policy_DoorKey_wrong,
        use_saved_envs=False,
        show_window=True,
    )
    print(sat)
    print()
    env = traces[0][len(traces[0]) // 2][0]
    env_list = list(load_list_from_pickle(env_name + "-envs.pkl")) + [env]

    sat, sat_trace_pairs = verify_policy_on_envs(
        env_name=env_name,
        env_list=env_list,
        policy=ground_truth_asp_register[env_name],
        show_window=True,
    )
    print(sat)
    print([s1 for s1, _ in sat_trace_pairs])
