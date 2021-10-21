import random
import itertools
import math
import cv2


def ransac_algo(corners, threshold, inliers, it):
    try:
        """
        This methods uses the RANSAC algorithm on set of points of image.
    
        Parameters
        ----------
        corners: array, Required
    
        threshold: float, Required
    
        inliers: Required
    
        it: int, Required
    
        Return list of list
        """
    
        maxpts = []
        passes = []
        success = []
        used = []
        endpoints = []
        for i in range(it):
            maxpts += [[0, 0]]
            passes += [[(0, 0)]]
            success += [0]
            used += [[]]
            endpoints += [[]]
        for j in range(17):
            items = random.sample(range(len(corners)), 2)
            endpoints[j] = [corners[items[0]], corners[items[1]]]
            passes[j] = [corners[items[0]], corners[items[1]]]
            line_dim = (corners[items[0]][0] - corners[items[1]][0])**2 + (corners[items[0]][1] - corners[items[1]][1])**2
            try:
                m = (corners[items[0]][0] - corners[items[1]][0]) / (corners[items[0]][1] - corners[items[1]][1])
            except:
                continue
            if m == 0:
                continue
            b = -m*corners[items[0]][1] + corners[items[0]][0]
            for k in range(len(corners)):
                cornerx = (corners[k][1] / m + corners[k][0] - b)/(m + 1/m)
                cornery = cornerx * m + b
                d = (corners[k][1] - cornerx) ** 2 + (corners[k][0] - cornery) ** 2
                if d <= threshold ** 2:
                    success[j] += 1
                    used[j] += [corners[k]]
                    first_dim = (corners[k][0] - endpoints[j][0][0])**2 + (corners[k][1] - endpoints[j][0][1])**2
                    second_dim = (corners[k][0] - endpoints[j][1][0])**2 + (corners[k][1] - endpoints[j][1][1])**2
                    if first_dim > line_dim or second_dim > line_dim:
                        if first_dim <= second_dim:
                            if first_dim > maxpts[j][0]:
                                maxpts[j][0] = first_dim
                                passes[j][0] = corners[k]
                        else:
                            if second_dim > maxpts[j][1]:
                                maxpts[j][1] = second_dim
                                passes[j][1] = corners[k]
                if success[j] >= inliers:
                    return [passes[j], used[j], endpoints[j]]
        winner = success.index(max(success))
        return [passes[winner] , used[winner], endpoints[winner]]
    except:
        print("Error in ransac_algo function")




def apply_ransac(img, corners, threshold = math.sqrt(3.84), inliers = 1000, features = 4, it = 17):
    try:
        """
        This methods apply the RANSAC algorithm on an image as per iteration value(it parameter) ,
        and creates an image.
    
        Parameters
        ----------
        img: array, Required
    
        corners: array, Required
    
        threshold: float, Optional
    
        inliers: int, Optional
        
        features: int, Optional
        
        it: int, Optional
        
        Return array
        """
        colors = list(itertools.product([0, 255], repeat = 3))
        color = random.sample(colors[1:-1], features + 1)
        for i in range(features):
            winner = ransac_algo(corners, threshold, inliers, it)
            img = cv2.line(img, (winner[0][0])[::-1], (winner[0][1])[::-1], color[i], 1)
            for j in range(len(winner[1])): 
                pt = winner[1][j]
                for row in range(3):
                    for column in range(3):
                        x = row - 1
                        y = column - 1
                        img[pt[0] + x][pt[1] + y] = color[i] if pt != winner[2][0] and pt != winner[2][1] else color[-1]
                corners.remove(pt)
        return img
    except:
        print("Error in apply_ransac function")