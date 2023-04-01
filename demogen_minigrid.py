#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
from copy import deepcopy

from commons_minigrid import Trace, demo_traces_to_pickle

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class ManualControl:
    def __init__(
        self,
        env_name: str,
        env: MiniGridEnv,
        agent_view: bool = False,
        window: Window = None,
        seed=None,
    ) -> None:
        self.env_name = env_name
        self.env = env
        self.agent_view = agent_view
        self.seed = seed
        self.demos = []
        self.current_demo = []

        if window is None:
            window = Window("minigrid - " + str(env.__class__))
        self.window = window
        self.window.reg_key_handler(self.key_handler)

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        self.window.show(block=True)
        demo_traces_to_pickle(self.demos, self.env_name)

    def step(self, action: Actions, act: str):
        s = deepcopy(self.env)
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        s_p = deepcopy(self.env)
        self.current_demo.append((s, act, s_p))

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.redraw()

    def redraw(self):
        frame = self.env.get_frame(agent_pov=self.agent_view)
        self.window.show_img(frame)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        if self.current_demo:
            self.demos.append(Trace(deepcopy(self.current_demo)))
            self.current_demo = []

        if hasattr(self.env, "mission"):
            print("Mission: %s" % self.env.mission)
            self.window.set_caption(self.env.mission)

        self.redraw()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.reset()
            self.window.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            " ": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "enter": Actions.done,
        }

        key_to_act_str = {
            "left": "left",
            "right": "right",
            "up": "forward",
            " ": "toggle",
            "pageup": "pickup",
            "pagedown": "drop",
            "enter": "done",
        }

        action = key_to_action[key]
        act = key_to_act_str[key]
        self.step(action, act)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-MultiRoom-N6-v0"
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

    env: MiniGridEnv = gym.make(args.env, tile_size=args.tile_size)

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    manual_control = ManualControl(
        args.env, env, agent_view=args.agent_view, seed=args.seed
    )
    manual_control.start()
