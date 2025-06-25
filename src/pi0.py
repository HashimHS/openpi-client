#!/usr/bin/env python
from typing import List
import rclpy
from rclpy.node import Node
from rclpy.context import Context
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint 
from rclpy.action import ActionClient 

# OpenPi
import dataclasses
import enum
import logging
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    UR5E = "ur5e"


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "0.0.0.0"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 5555
    # API key to use for the server.
    api_key: str | None = None
    # Number of steps to run the policy for.
    num_steps: int = 20
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Environment to run the policy in.
    env: EnvMode = EnvMode.UR5E


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def PiTest(args: Args) -> None:
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
        EnvMode.UR5E: _random_observation_ur5e,
    }[args.env]

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send a few observations to make sure the model is loaded.
    for _ in range(2):
        policy.infer(obs_fn())

    timing_recorder = TimingRecorder()

    for _ in tqdm.trange(args.num_steps, desc="Running policy"):
        inference_start = time.time()
        action = policy.infer(obs_fn())
        timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
        for key, value in action.get("server_timing", {}).items():
            timing_recorder.record(f"server_{key}", value)
        for key, value in action.get("policy_timing", {}).items():
            timing_recorder.record(f"policy_{key}", value)

    timing_recorder.print_all_stats()

    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)

class RGBListener(Node):
    def __init__(self, topic='/color/image_color', node_name='rgb_listener'):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.image = None
        self.rate = self.create_rate(15)
        default_qos_profile = QoSProfile(depth=5)
        self.subscriber = self.create_subscription(Image, topic=topic, callback=self.callback, qos_profile=default_qos_profile)

    def callback(self, data):
        if self.image is not None: 
            return
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except:
            pass

    def get(self):
        while self.image is None:
            self.rate.sleep()
        return self.image

class joint_states_listener(Node):
    def __init__(self, topic='/joint_states', node_name='joint_states_listener'):
        super().__init__(node_name)
        self.joint_states = None
        default_qos_profile = QoSProfile(depth=5)
        self.subscriber = self.create_subscription(JointState, topic=topic, callback=self.callback, qos_profile=default_qos_profile)

    def callback(self, data):
        if self.joint_state is not None: 
            return
        self.joint_states = data

    def get(self):
        while self.joint_states is None:
            rclpy.spin_once(self)
        return self.joint_states

class PiService(Node):
    def __init__(self) -> None:
        super().__init__("pi_service")
        args = Args()
        self.logger = self.get_logger()
        default_qos_profile = QoSProfile(depth=5)
        self.wrist_subscriber = RGBListener("color/image_wrist", node_name='wrist_rgb_listener')
        self.base_subscriber = RGBListener("color/image_base", node_name='base_rgb_listener')
        self.joint_state_subscriber = joint_states_listener(topic='/joint_states')
        self.gripper = np.zeros((1,), dtype=np.float32)  # Placeholder for gripper state
        # self.prompt_subscriber = self.node.create_subscription(String, topic='/prompt', callback=self.prompt_callback, qos_profile=default_qos_profile)
        self.logger.info("Waiting for the policy server to be ready...")
        self.logger.info(f"Connecting to server at {args.host}:{args.port} with API key: {args.api_key}")
        self.policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )        
        self.logger.info(f"Server metadata: {self.policy.get_server_metadata()}")

        # Send a few observations to make sure the model is loaded.
        self.logger.info("Sending initial observations to make sure the model is loaded.")
        obs_fn = _random_observation_ur5e
        for _ in range(2):
            results = self.policy.infer(obs_fn())
        self.logger.info(f"Policy results: {results}")

        self.timing_recorder = TimingRecorder()
        
        # Create an action client for the arm joint trajectory controller.
        self.logger.info("Waiting for the joint trajectory controller action server...")
        self.controller = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
        self.controller.wait_for_server()
        self.logger.info("Joint trajectory controller action server is ready.")
        
        self.run(prompt="pick up the object")

    def obs_fun(self, prompt) -> dict:
        """Generate a random observation for the UR5E environment."""
        obs = {
            "joints": np.array(self.joint_state_subscriber.get().position),
            "gripper": self.gripper,
            "base_rgb": self.base_subscriber.get(),
            "wrist_rgb": self.wrist_subscriber.get(),
            "prompt": prompt,
        }
        self.joint_commands = obs["joints"].copy()
        return obs

    def run(self, prompt: str = "do something") -> None:
        """Run the PiService with the given prompt."""
        while rclpy.ok():

            # Run the policy inference.
            inference_start = time.time()
            actions = self.policy.infer(self.obs_fun(prompt))["actions"]
            self.logger.info(f"Policy actions: {actions}")

            # Create a joint command message.
            joint_command = FollowJointTrajectory.Goal()
            joint_command.trajectory = JointTrajectory()
            joint_command.trajectory.joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
            joint_command.trajectory.points = [JointTrajectoryPoint(positions=actions, time_from_start=rclpy.duration.Duration(seconds=1.0))]
            self.logger.info(f"Joint command: {joint_command}")
            # Send the joint command to the action server.
            self.controller.send_goal_async(joint_command)
            self.controller.get_result_async()
            # self.timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
            # for key, value in actions.get("server_timing", {}).items():
            #     self.timing_recorder.record(f"server_{key}", value)
            # for key, value in actions.get("policy_timing", {}).items():
            #     self.timing_recorder.record(f"policy_{key}", value)

def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def _random_observation_ur5e() -> dict:
    """Creates a random input example for the UR5E policy."""
    return {
        "joints": np.random.rand(6),
        "gripper": np.random.rand(1),
        "base_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def main():
    rclpy.init()
    service = PiService()
    rclpy.spin(service)
    rclpy.shutdown()
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()