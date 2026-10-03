# Adapted from Pollen Robotics' Reachy Mini quickstart (https://huggingface.co/docs/reachy_mini/SDK/quickstart).
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini(media_backend="no_media") as mini:
    print("Connected to Reachy Mini!")

    print("Wiggling antennas...")
    mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
    mini.goto_target(antennas=[0, 0], duration=0.5)

    print("Looking left, right, and nodding...")
    mini.goto_target(head=create_head_pose(yaw=20, degrees=True), duration=1.0)
    mini.goto_target(head=create_head_pose(yaw=-20, degrees=True), duration=1.0)
    mini.goto_target(head=create_head_pose(pitch=10, degrees=True), duration=0.6)
    mini.goto_target(head=create_head_pose(), duration=1.0)

    print("Done!")
