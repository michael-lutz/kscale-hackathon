# UBE: Universal Vision-Based End-Effector for Robotic Manipulation

To quickly calibrate camera:
1. Print out `assets/chessboard.pdf` and ensure squares are exactly 1x1 inches
2. Run:
"""bash
python3 scripts/aruco/calibrate_camera.py
"""

To track and visualize wrist motion run:
"""bash
python3 ube/scratch/claw_tracker_scratch.py
"""
