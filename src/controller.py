from typing import List
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.duration import Duration

class RobotController(Node):
    def __init__(self, topic='/joint_trajectory_controller/follow_joint_trajectory'):
        super().__init__('robot_controller')
        self.action_client = ActionClient(self, FollowJointTrajectory, topic)
        self.action_client.wait_for_server()
        self.command_listener = self.create_subscription(
            FollowJointTrajectory,
            topic,
            self.command_callback,
            10
        )

    def send_joint_command(self, joint_names: List[str], positions: List[float], time_from_start: float):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = joint_names
        point = JointTrajectoryPoint(positions=positions, time_from_start=rclpy.duration.Duration(seconds=time_from_start))
        goal.trajectory.points.append(point)
        return self.action_client.send_goal_async(goal)
    
    def get_result(self, future):
        """Get the result of the action."""
        if future.done():
            result = future.result()
            
    def cancel_all_goals(self):
        """Cancel all goals."""
        self.action_client.cancel_all_goals()
        
    def command_callback(self, msg: FollowJointTrajectory):
        """Handle incoming joint trajectory commands."""
        # Here you can process the command further or send it to the robot
        self.new_action = True
        self.cancel_all_goals()
        future = self.send_joint_command(msg.trajectory.joint_names, msg.trajectory.points[0].positions, msg.trajectory.points[0].time_from_start.nanoseconds / 1e9)

if __name__ == '__main__':
    rclpy.init()
    controller = RobotController()
    rclpy.spin(controller)