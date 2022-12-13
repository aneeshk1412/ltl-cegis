#!/usr/bin/env python3

import pickle
import gymnasium as gym

from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class ManualCheck:
    def __init__(
        self,
        env: MiniGridEnv,
        env_name: str,
        action_selection_policy,
        agent_view: bool = False,
        window: Window = None,
        seed=None,
    ) -> None:
        self.env = env
        self.env_name = env_name
        self.action_selection_policy = action_selection_policy

        self.agent_view = agent_view
        self.seed = seed

        if window is None:
            window = Window("minigrid - " + str(env.__class__))
        self.window = window
        self.window.reg_key_handler(self.key_handler)

    def start(self):
        self.reset(self.seed)
        self.window.show(block=True)

    def step(self, action: MiniGridEnv.Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

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

        if hasattr(self.env, "mission"):
            print("Mission: %s" % self.env.mission)
            self.window.set_caption(self.env.mission)

        self.redraw()

    def save_env_to_dataset(self):
        with open(self.env_name + ".pkl", "ab+") as f:
            pickle.dump(self.env, f)

    def key_handler(self, event):
        key: str = event.key

        if key == "escape":
            self.window.close()
            return
        if key == "backspace":
            self.reset()
            return
        if key == "j":
            self.save_env_to_dataset()
            return

        if self.action_selection_policy:
            key = self.action_selection_policy(self.env)

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
        self.step(action)


if __name__ == "__main__":
    import argparse
    from dsl_minigrid import feature_register
    from asp_minigrid import ground_truth_asp_register, action_selection_policy_decision_tree

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        help="gym environment to load",
        default="MiniGrid-DoorKey-16x16-v0",
    )
    parser.add_argument(
        "--ground-truth",
        default=False,
        help="whether to run the ground truth model for this environment",
        action="store_true",
    )
    parser.add_argument(
        "--manual",
        default=False,
        help="whether to run manually on this environment",
        action="store_true",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        help="Model File to run, if none of --manual or --ground-truth flag is specified",
        default="DT.model",
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

    env: MiniGridEnv = gym.make(args.env_name, tile_size=args.tile_size)

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    if args.manual:
        action_selection_policy = None
    elif args.ground_truth:
        action_selection_policy = ground_truth_asp_register[args.env_name]
    else:
        with open(args.model_file, "rb") as f:
            aspmodel = pickle.load(f)
        action_selection_policy = lambda env: action_selection_policy_decision_tree(env, aspmodel, feature_register[args.env_name])

    manual_check = ManualCheck(
        env,
        args.env_name,
        action_selection_policy,
        agent_view=args.agent_view,
        seed=args.seed,
    )
    manual_check.start()
