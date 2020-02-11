#!/usr/bin/python3

# basic steps
#
# convert the frame to a 2D flattened perspective to make z optical flow easier
# get the optical flow for the features from one frame to the next
# remove outliers from the list of flows and get the mean of the flows for each frame
# get a rolling average of the results for the video and least square fit to the ground truth to obtain a scale
# multiply the scale by the smoothed averages to get predicted speed at each frame

import sys
import cv2
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error 


# LK flow and feature detection params 
#          winSize = area in px to detect motion
#          maxLevel = max pyramid level to attempt detection (higher level = higher highest resolution)
#          criteria = same in every example

lkParams = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

# Shi-Tomasi corner derection params
#           maxCorners = max number of features to track
#           qualityLevel = threshold to be considered a corner
#           minDistance = minimum distance
#           blockSize = size of the window to run corner detection inside of

featureParams = dict( maxCorners = 200,
                       qualityLevel = 0.1,
                       minDistance = 5,
                       blockSize = 5 )

class SpeedEstimator:
    
    def __init__(self, gt_txt):
        self.gt = None
        if gt_txt is not None:
            with open(gt_txt) as ground_truth:
                self.gt = ground_truth.read().splitlines()
        self.scale_factor = 1
    
    def rollingavg(self, lst, window):
        ret = np.zeros_like(lst)

        for i in range(len(lst)):
            idx1 = max(0, i - (window - 1) // 2)
            idx2 = min(len(lst), i + (window - 1) // 2 + (2 - (window % 2)))

            ret[i] = np.mean(lst[idx1:idx2])

        return ret

    def remove_outliers(self, arr):
        # removes outliers using IQR rule, with a factor of 1.5
        x = np.array(arr)
        uq = np.percentile(x, 75)
        lq = np.percentile(x, 25)
        iqr = (uq - lq)*1.5
        quartiles = (lq - iqr, uq + iqr) 
        return x[np.where((x >= quartiles[0]) & (x <= quartiles[1]))]
        
    def opticalflow(self, frame, frame_prev, points_prev):
        # return the u and v values 
        points, _st, _err = cv2.calcOpticalFlowPyrLK(frame_prev, frame, points_prev, None, **lkParams)
        return (points - points_prev).reshape(-1, 2)

    def train(self, video, train=True):
        cap = cv2.VideoCapture(video)
        means = []
        raw_preds = []
        self.frame = 0
        self.features_prev = None
        self.img_prev = None
        self.kpts_prev = None
        
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break
            bw_frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # transform perspective to 2D
            pts1 = np.float32([[270, 240], [350, 240], [0, 365], [640, 365]])
            pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.GaussianBlur(cv2.warpPerspective(bw_frames, matrix, (640, 480)), (3, 3), 0)

            # show the nice, unaltered frame
            cv2.imshow("video", frame)
            frame = cv2.warpPerspective(frame, matrix, (640, 480))

            # get optical flow after we have data for the first frame 
            if self.features_prev is not None:

                flow = self.opticalflow(img, self.img_prev, self.features_prev)           
                zvals = []

                for u, v in flow:  
                    # only return flow for objects going y+  
                    if (v > 0) and (abs(u) + abs(v)) < 100:          
                        zvals.append(v)
                    else:
                        zvals.append(0)

                # now remove the outliers and get average flow between frames
                zvals = self.remove_outliers(zvals)
                raw_preds = [n for n in zvals if n>0]              
                means.append((np.mean(raw_preds) if len(raw_preds) else 0))

            features = cv2.goodFeaturesToTrack(img, **featureParams)

            if self.features_prev is None:
                print("skipped frame %s" % self.frame)
                means.append(0)
                kpts = None
            else:                
                kpts = [cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=10) for p in self.features_prev]
                # show the transformed frame with the features overlayed
                cv2.drawKeypoints(frame, kpts, frame, color=(0, 0, 255))
                cv2.drawKeypoints(frame, self.kpts_prev, frame, color=(0, 255, 0))
                # draw lines connecting the features
                linecoords = [[tuple(i) for i in self.features_prev.reshape(-1,2)],[tuple(i) for i in features.reshape(-1,2)]]
                for f1, f2 in zip(linecoords[0], linecoords[1]):
                    if (abs(f2[0]-f1[0]) < 20):
                        cv2.line(frame, f2, f1, (255,0,0), 2)
                    
            cv2.imshow("transform", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows
                break

            self.features_prev = features
            self.kpts_prev = kpts
            self.img_prev = img
            self.frame += 1

        # smooth out the means to reflect the motion of a real car and take care of any outliers in this set
        predicted_speeds = self.rollingavg(means, 80)
        print(self.frame, len(means))

        if train:   
            # find the scale factor by least square fitting           
            self.scale_factor = np.linalg.lstsq(np.array(predicted_speeds).reshape(-1,1),
                                    np.array(self.gt[:len(predicted_speeds)]).reshape(-1,1))[0][0][0]
            print("ESTIMATED SCALE: %s" % self.scale_factor)
            

        return self.scale_factor, predicted_speeds

def calculate_mse(gtruth, test):
    with open(gtruth) as gt:
        truth = gt.read().splitlines()
    with open(test) as est:
        estimate = est.read().splitlines()
    # calculate the MSE
    mse = mean_squared_error([float(i) for i in truth[:len(estimate)]], [float(i) for i in estimate])
    return mse

def main():
    gt_txt = './data/train.txt'
    video = './data/train.mp4'
    video_test = './data/test.mp4'
    out1_txt = 'train_output.txt'
    out2_txt = 'test.txt'

    # get the scale factor
    se = SpeedEstimator(gt_txt)
    scale, out = se.train(video)

    # write output to a file
    outfile = open(out1_txt,"w+")
    for i in out:
        outfile.write("%s\n" % (i * scale) )

    # test our other video
    _, out2 = se.train(video_test, train=False)
    outfile2 = open(out2_txt,"w+")
    for i in out2:
        outfile2.write("%s\n" % (i * scale))

if __name__ in '__main__':
    main()
    mse = calculate_mse('./data/train.txt', 'train_output.txt')
    print("mean square error for training data: %s" % mse)