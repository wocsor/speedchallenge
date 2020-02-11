Welcome to the comma.ai Programming Challenge!
======

Your goal is to predict the speed of a car from a video.

- data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
- data/train.txt contains the speed of the car at each frame, one speed on each line.
- data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.

Deliverable
-----

Your deliverable is test.txt. E-mail it to givemeajob@comma.ai, or if you think you did particularly well, e-mail it to George.

Evaluation
-----

We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.

Twitter
------

<a href="https://twitter.com/comma_ai">Follow us!</a>

## Requirments
opencv
numpy
sklearn

## Usage
place `test.mp4`, `train.mp4`, and `train.txt` inside of the `data` folder and run the script with ./speed_estimator.py

for test.mp4 this program achieves a mse of 4.157566952137437

## How it works
1) get a 2D perspective (birds-eye view) to compensate for perspective projection / radial expansion from FoE
2) track features on the new, top-down view
3) get average optical flow of features moving primarily toward the camera, removing any outliers
4) smooth the average flows using a rolling average
6) least-square fit the average flows to the ground truth to obtain a scale factor
7) multiply the scale factor to the average flows to get predicted speeds
8) repeat for the test video, skipping step 6 to obtain predicted speed at each frame
9) the estimated scale and mse are printed in the console


