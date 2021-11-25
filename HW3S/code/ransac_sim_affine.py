import cv2
import numpy as np
import random
from get_matches import *

def ransac_sim_affine(simA, simB):
    img1, img2, kpts1, kpts2, matches = get_matches(simA, simB)
    tolerance = 1 # used when comparing similarity matrices of match pairs
    consensus_set = []
    best_sim = []
    # find consensus of translation between a random keypoint and the rest for
    # a number of times to find the best match regarding translation
    for i in range(100):
        idxs = random.sample(range(len(matches)), 3)
        # calc similarity between kp1, kp2 and kp3
        kp11 = kpts1[matches[idxs[0]].queryIdx]
        kp12 = kpts2[matches[idxs[0]].trainIdx]
        kp21 = kpts1[matches[idxs[1]].queryIdx]
        kp22 = kpts2[matches[idxs[1]].trainIdx]
        kp31 = kpts1[matches[idxs[2]].queryIdx]
        kp32 = kpts2[matches[idxs[2]].trainIdx]
        #  A = np.array([[kp11.pt[0], kp11.pt[1], 1, 0, 0, 0],
                      #  [0, 0, 0, kp11.pt[0], kp11.pt[1], 1],
                      #  [kp21.pt[0], kp21.pt[1], 1, 0, 0, 0],
                      #  [0, 0, 0, kp21.pt[0], kp21.pt[1], 1],
                      #  [kp31.pt[0], kp31.pt[1], 1, 0, 0, 0],
                      #  [0, 0, 0, kp31.pt[0], kp31.pt[1], 1]])
        #  b = np.array([kp12.pt[0], kp12.pt[1], kp22.pt[0], kp22.pt[1],
                      #  kp32.pt[0], kp32.pt[1]])
        #  Sim,_,_,_ = np.linalg.lstsq(A, b)
        #  Sim = Sim.reshape((2, 3))
        pts1 = np.float32([[kp11.pt[0], kp11.pt[1]],
                           [kp21.pt[0], kp21.pt[1]],
                           [kp31.pt[0], kp31.pt[1]]])
        pts2 = np.float32([[kp12.pt[0], kp12.pt[1]],
                           [kp22.pt[0], kp22.pt[1]],
                           [kp32.pt[0], kp32.pt[1]]])
        Sim = cv2.getAffineTransform(pts1, pts2)
        temp_consensus_set = []
        for j in range(len(matches)-2):
            kp11 = kpts1[matches[j].queryIdx]
            kp12 = kpts2[matches[j].trainIdx]
            kp21 = kpts1[matches[j+1].queryIdx]
            kp22 = kpts2[matches[j+1].trainIdx]
            kp31 = kpts1[matches[j+2].queryIdx]
            kp32 = kpts2[matches[j+2].trainIdx]
            pts1 = np.float32([[kp11.pt[0], kp11.pt[1]],
                               [kp21.pt[0], kp21.pt[1]],
                               [kp31.pt[0], kp31.pt[1]]])
            pts2 = np.float32([[kp12.pt[0], kp12.pt[1]],
                               [kp22.pt[0], kp22.pt[1]],
                               [kp32.pt[0], kp32.pt[1]]])
            Sim2 = cv2.getAffineTransform(pts1, pts2)
            if (np.array(np.abs(Sim-Sim2)) < tolerance).all():
                temp_consensus_set.append(j)
                temp_consensus_set.append(j+1)
                temp_consensus_set.append(j+2)
        if len(temp_consensus_set) > len(consensus_set):
            consensus_set = temp_consensus_set
            best_sim = Sim
    consensus_matches = np.array(matches)[consensus_set]
    matched_image = np.array([])
    # draw matches with biggest consensus
    matched_image = cv2.drawMatches(img1, kpts1, img2, kpts2, consensus_matches,
                                    flags=2, outImg=matched_image)
    print('Best match:\nSim=\n%s\n with consensus=%d or %d%%'%(
        best_sim, len(consensus_set)/3, 100*len(consensus_set)/len(matches)))
    return matched_image, best_sim
