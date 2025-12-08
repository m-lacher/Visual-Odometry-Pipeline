# Visual-Odometry-Pipeline
Optional mini Project of the course "Vision Algorithms for Mobile Robotics"

# Setup Instructions
We are using conda to manage the dependencies as suggested in the pdf.
1) download conda
2) conda env create -f environment.yml
3) conda activate vo_env

# TODOs
- Cleanup main file
- Improve precision and speed of pipeline. Potential work to be done:
    1) Improve data handling (replace lists with numpy array (e.g. Map_Points), lots of transforms and list manipulations
    2) Tune Parameters
    3) Improve plotting efficiency
