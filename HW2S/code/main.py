"""
Name: Kunal Goyal
CS558
Computer Vision
"""


import numpy as np
import random
import itertools
import math
import cv2
import filterop
import ransac
import hough



if __name__ == "__main__":

    try:
        """Filtering variable and corrosponding values.
        g: gaussian
        sobelFilHorizontal: Horizontal Sobel Filter values
        sobelFilVerticle: Verticle Sobel Filter values
        """
        g = [[0.077847, 0.123317, 0.077847], 
                    [0.123317, 0.195346, 0.123317], 
                    [0.077847, 0.123317, 0.077847]]
        sobelFilHorizontal = [[1, 2, 1], 
                              [0, 0, 0], 
                              [-1, -2, -1]]
        sobelFilVerticle = [[1, 0, -1], 
                [2, 0, -2], 
                [1, 0, -1]]
        
        
        # Get the image
        img = cv2.imread("road.png", 0)
        arr = filterop.updated_arr(img)
    
        gfilImg = filterop.apply_filter(g, arr)
        h = filterop.apply_filter(sobelFilHorizontal, gfilImg[0])  
        v = filterop.apply_filter(sobelFilVerticle, gfilImg[0])

        
        """Apply filter for Horizontal and Vertical sobel
           Get Key points in the image using a Hessian detector
        cord1_xx: for XX cordinates
        cord1_yy: for YY cordinates
        cord1_xy: for XY cordinates
        cord1_yx: for YX cordinates
        """
        cord1_xx = filterop.apply_filter(sobelFilHorizontal, h[0])
        cord1_yy = filterop.apply_filter(sobelFilVerticle, v[0])
        cord1_xy = filterop.apply_filter(sobelFilVerticle, h[0])
        cord1_yx = filterop.apply_filter(sobelFilHorizontal, v[0])
        
        edges = filterop.overlay_image(h[0], v[0])
        suppress_edges = filterop.sup_nm(edges[0], h[0], v[0], "edges")
        threshold_edges = filterop.threshold(suppress_edges[0], 175, 60)
   
        hess_matrix = filterop.hes_matrix(cord1_xx[0], cord1_yy[0], cord1_xy[0], cord1_yx[0], 175000)
        hess_threshold = filterop.sup_nm(hess_matrix[0], h[0], v[0], "corners")
        updated_hess_matrix = filterop.overlay_image(hess_threshold[0], threshold_edges / 4, bg=True) 
        
        colored = (updated_hess_matrix[1]).copy()
        colored = cv2.cvtColor(updated_hess_matrix[1], cv2.COLOR_GRAY2RGB)
        
        """
        Applying RANSAC 
        Run the RANSAC algorithm on the key points to find the 4 best lines 
        """
        ransac_image = colored.copy()
        ransac_image = ransac.apply_algo(ransac_image, filterop.corners_to_list(hess_threshold[1]), itr = 20)
        
        """
        Hough Transformation with the hess_threshold values
        """
        Hough_transform  = colored.copy()
        Hough_transform  = hough.apply_hough(Hough_transform, filterop.corners_to_list(hess_threshold[1]), angle=45)
        
        
        """
        Saving all output images
        """
        cv2.imwrite("gaussian_filter.png", gfilImg[1])
        cv2.imwrite("h_sobel_filter.png", h[1])
        cv2.imwrite("v_sobel_filter.png", v[1])
        cv2.imwrite("cord1.png", cord1_xx[1])
        cv2.imwrite("cord2.png", cord1_yy[1])
        cv2.imwrite("cord3.png", cord1_xy[1])
        cv2.imwrite("cord4.png", cord1_yx[1])
        cv2.imwrite("edges_with_no_supress.png", edges[1])
        cv2.imwrite("edges_with_supress.png", suppress_edges[1])
        cv2.imwrite("edges_threshold.png", threshold_edges)
        cv2.imwrite("corners.png", hess_matrix[1])
        cv2.imwrite("corners_threshold.png", hess_threshold[1])
        cv2.imwrite("updated_corners.png", updated_hess_matrix[1])
        cv2.imwrite("ransac.png", ransac_image)
        cv2.imwrite("hough_image.png", Hough_transform)
    except Exception as e:
        print(e)