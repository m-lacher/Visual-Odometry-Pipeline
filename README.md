# Visual-Odometry-Pipeline
Optional mini Project of the course "Vision Algorithms for Mobile Robotics"

# Setup Instructions
We are using conda to manage the dependencies as suggested in the pdf.
1) download conda
2) conda env create -f environment.yml
3) conda activate vo_env
4) python main.py

# Datasets
Our github repository only comes with our own dataset included. We think this is so that you dont download the same dataset many times on your machines.
Please put the datasets that we recieved in the datasets folder so that the kitti, malaga, and parking have their own folders (we didnt change them)
To run different datasets change the number in the main.py file ds = #  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
Finally, for the parking dataset we changed the parameters of our algorithm a little bit because it seems it was made in simulation. The repo comes configured for real world datasets The parking dataset will still run but not optimally like in the video.

# Machine used to make videos
i7 13700K
3400 MHz
32GB ram
max threads: 13

# Youtube Links
[kitti dataset](https://www.youtube.com/watch?v=RO2LaTvn-aU) \
[malaga dataset](https://www.youtube.com/watch?v=eqegAThBELw) \
[parking dataset](https://www.youtube.com/watch?v=GJ_gRpAT5MY) \
[our own dataset](https://youtu.be/kjrYvDRPZwk)

# Authors
**Piotr Maciej Wojtaszewski** \
**Gian-Andrin Coolen** \
**Markus Lacher** 

