#!/usr/bin/env python3

import pickle
from copy import deepcopy
from typing import Callable, List

from utils import intersperse
from dsl_minigrid import feature_register, header_register

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv


def implies(b1, b2):
    return not b1 or b2


class Runner(object):
    def __init__(
        self,
        renv: MiniGridEnv | None,
        env_name: str,
        policy: Callable[[MiniGridEnv], str] | None = None,
        seed: int | None = None,
        max_steps: int = 100,
        window: Window | None = None,
        block: bool = False,
        num_rruns: int | None = None,
        env_list: List[MiniGridEnv] = [],
    ) -> None:
        self.renv = renv
        self.env = None
        self.env_name = env_name
        self.env_list = env_list
        self.policy = policy
        self.obs_func = feature_register[self.env_name]
        self.headers = header_register[self.env_name]
        self.window = window
        self.block = block
        self.seed = seed
        self.max_steps = max_steps
        self.num_rruns = num_rruns
        if self.renv is None:
            self.num_runs = len(env_list)
        elif self.num_rruns is None:
            self.num_runs = None
        else:
            self.num_runs = len(env_list) + self.num_rruns
        self.cur_run = 0

        assert implies(not self.block, self.policy is not None)
        assert implies(self.block, self.window is not None)
        assert implies(not self.block, self.num_runs is not None)
        assert implies(self.num_runs is None, self.renv is not None)

        if self.block:
            self.window.reg_key_handler(self.key_handler)

        self.traces = []
        self.sat: bool = True
        self.key_to_action = {
            "left": MiniGridEnv.Actions.left,
            "right": MiniGridEnv.Actions.right,
            "up": MiniGridEnv.Actions.forward,
            " ": MiniGridEnv.Actions.toggle,
            "pageup": MiniGridEnv.Actions.pickup,
            "pagedown": MiniGridEnv.Actions.drop,
            "enter": MiniGridEnv.Actions.done,
        }

    def run(self):
        raise NotImplementedError()

    def run_internal(self) -> None:
        self.reset()
        if self.window:
            self.window.show(block=self.block)
        if not self.block:
            while self.step(None):
                pass
            if self.window:
                self.window.close()

    def stopping_cond(self) -> bool:
        raise NotImplementedError()

    def get_next_env(self) -> MiniGridEnv | None:
        if self.num_rruns is None:
            if self.cur_run <= len(self.env_list):
                self.env_list[self.cur_run - 1].soft_reset(
                    seed=self.seed, max_steps=self.max_steps
                )
                return self.env_list[self.cur_run - 1]
            elif self.renv is not None:
                self.renv.reset(seed=None)
                self.renv.soft_reset(seed=self.seed, max_steps=self.max_steps)
                return self.renv
            else:
                return None
        else:
            if self.cur_run <= self.num_rruns:
                self.renv.reset(seed=None)
                self.renv.soft_reset(seed=self.seed, max_steps=self.max_steps)
                return self.renv
            elif self.cur_run <= self.num_rruns + len(self.env_list):
                self.env_list[self.cur_run - self.num_rruns - 1].soft_reset(
                    seed=self.seed, max_steps=self.max_steps
                )
                return self.env_list[self.cur_run - self.num_rruns - 1]
            else:
                return None

    def reset(self) -> bool:
        self.traces.append([])
        self.cur_run += 1
        self.env = self.get_next_env()
        if self.env is not None:
            self.redraw()
        assert implies(self.env is None, not self.stopping_cond())
        if not self.stopping_cond() and self.block:
            self.window.close()
        return self.stopping_cond()

    def redraw(self) -> None:
        if self.window:
            frame = self.env.get_frame(agent_pov=False)
            state = self.obs_func(self.env)
            caption = "    ".join(
                intersperse(
                    [self.headers[i] for i in range(len(state)) if state[i]], "\n", 2
                )
            )
            self.window.set_caption(caption)
            self.window.show_img(frame)

    def save_env(self) -> None:
        with open(self.env_name + "-envs.pkl", "ab+") as f:
            pickle.dump(self.env, f)

    def process_trace(self, sat: bool) -> None:
        raise NotImplementedError()

    def step(self, key: str) -> bool:
        if self.policy:
            key = self.policy(self.env)

        state = deepcopy(self.env)
        action = self.key_to_action[key]

        _, reward, terminated, truncated, _ = self.env.step(action)

        next_state = deepcopy(self.env)
        self.traces[-1].append(
            (state, self.obs_func(state), key, next_state, self.obs_func(next_state))
        )

        if truncated:
            self.sat = False
            self.process_trace(False)
            return self.reset()
        elif terminated and reward < 0:
            self.sat = False
            self.process_trace(False)
            return self.reset()
        elif terminated:
            self.process_trace(True)
            return self.reset()
        else:
            self.redraw()
            return True

    def key_handler(self, event=None) -> bool:
        key = event.key
        if key == "escape":
            self.window.close()
            return
        if key == "backspace":
            self.reset()
            return
        if key == "e":
            self.save_env()
            return
        if key == "d":
            self.traces[-1] = []
            return

        self.step(key)
