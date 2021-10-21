import numpy as np
import random
import itertools
import math
import cv2

def non_max_supresn(arr, horizontal, vertical, mode):
    try:
        """
        This method supresses pixel intensities based on the neighborhood pixel, 
            
        Parameters
        ----------
        arr: array, Required
    
        horizontal:array, Required
    
        vertical:array, Required
    
        mode : str , Required
    
        return list
        """
    
        canvas = arr.copy()
        img = arr.copy()
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if mode == "edges":
                    angle = math.atan2(vertical[i][j], horizontal[i][j]) 
                    canvas[i][j] = arr[i][j]
                    if i == 0 or j == 0 or i == len(arr) - 1  or j == len(arr[i]) - 1:
                        canvas[i][j] = 0
                    elif (angle >=  -1*math.pi/8 and angle <= math.pi / 8) or (angle > 7*math.pi/8 and angle <= -7*math.pi/8):
                        if arr[i][j] <= arr[i][j+1] or arr[i][j] <= arr[i][j-1]:
                            canvas[i][j] = 0
                    elif (angle < -1*math.pi/8 and angle >= -3*math.pi/8) or (angle > math.pi/8 and angle <= 3*math.pi/8):
                        if arr[i][j] <= arr[i+1][j+1] or arr[i][j] <= arr[i-1][j-1]:
                            canvas[i][j] = 0
                    elif (angle < -3*math.pi/8 and angle >= -5*math.pi/8) or (angle > 3*math.pi/8 and angle <= 5*math.pi/8):
                        if arr[i][j] <= arr[i+1][j] or arr[i][j] <= arr[i-1][j]:
                            canvas[i][j] = 0
                    elif (angle < -5*math.pi/8 and angle >= -7*math.pi/8) or (angle > 5*math.pi/8 and angle <= 7*math.pi/8):
                        if arr[i][j] <= arr[i+1][j-1] or arr[i][j] <= arr[i-1][j+1]:
                            canvas[i][j] = 0
                    else:
                        canvas[i][j] = 0 
                    
                    
                    img[i][j] = 0 if canvas[i][j] < 0 else 255 if canvas[i][j] > 255 else canvas[i][j]
                        
                elif mode == "corners":
                    canvas[i][j] = arr[i][j]
                    if (i == 0 or j == 0 or i == len(arr) - 1 or j == len(arr[i]) - 1):
                        canvas[i][j] = 0
                    elif not (arr[i][j] > arr[i+1][j+1] and arr[i][j] > arr[i-1][j-1] and arr[i][j] > arr[i+1][j-1] and arr[i][j] > arr[i-1][j+1] and arr[i][j] > arr[i][j+1] and arr[i][j] > arr[i][j-1] and arr[i][j] > arr[i+1][j] and arr[i][j] > arr[i-1][j]):
                        canvas[i][j] = 0
    
                    img[i][j] = 0 if canvas[i][j] < 0 else  255 if canvas[i][j] > 255 else  canvas[i][j]
                
                        
        return [canvas, img]
    except:
        print("Error in non_max_supresn fn")

def threshold(canvas,min_value,max_value):
    try:
        
        """
        This method used to set pixel value to zero when it less than min_value,
        and if it is greater than min_value and less than max_value then set to 125 else 255.
            
        Parameters
        ----------
        canvas: , Required
    
        min_value:int, Required
    
        max_value:int, Required
    
        return canvas
        """
        canvas_len=len(canvas)
        for i in range(canvas_len):
            canvas_item_len=len(canvas[i])
            for j in range(canvas_item_len):
                canvas[i][j] = 0 if canvas[i][j] < min_value else  125 if canvas[i][j] < max_value else 255
        return canvas
    except:
        print("Error in threshold function")

def overlay_image(item1, item2, bg = False):
    try:
        """
        This method used to overlays images based on gradient value.
        Parameters
        ----------
        item1:array , Required
    
        item2:array, Required
    
        bg:boolean, Optional
    
        return tuple of item1 and image
        """
        img = item1.copy()
        for i in range(len(item1)):
            for j in range(len(item1[i])):
                if not bg:
                    val = math.sqrt(item1[i][j] ** 2 + item2[i][j] ** 2)
                    item1[i][j] = val
                    img[i][j] = 0 if val < 0 else 255 if val > 255 else  val
                else:
                    if item2[i][j] > item1[i][j]:
                        item1[i][j] = item2[i][j]
                    
                    img[i][j] = 0 if item1[i][j] < 0 else 255 if item1[i][j] > 255 else item1[i][j]
        return (item1, img)
    except:
        print("Error in overlay_image function")


def hes_matrix(xxcord, yycord, xycord, yxcord, threshold):
    try:
        """
        This method used to hessian matrix to detect corners
    
        Parameters
        ----------
        xxcord:array , Required
    
        yycord:array, Required
    
        xycord:array , Required
    
        yxcord:array, Required
    
        threshold:float, Required
    
        return list of list item
        """
        arr = xxcord.copy()
        img = xxcord.copy()
        for i in range(len(xxcord)):
            for j in range(len(xxcord[i])):
                det = xxcord[i][j]*yycord[i][j]-xycord[i][j]*yxcord[i][j]
                trace = xxcord[i][j] + yycord[i][j]
                r = det - .06*(trace**2)
                arr[i][j] = r
                if r > threshold:
                    img[i][j] = 255
                else:
                    arr[i][j] = 0
                    img[i][j] = 0
        return [arr, img]
    except:
        print("Error in hes_matrix fn")

def updated_arr(img):
    try:
        """
        This method create a array of a image
        Parameters
        ----------
        img:array, Required
    
        return list 
        """
        arr_item = []
        for i in range(len(img)):
            arr_item += [[]]
            for j in range(len(img[i])):
                arr_item[i] += [img[i][j]]
        arr_item = np.array(arr_item, dtype="float32")
        return arr_item
    except:
        print("Error updated_arr in Function")

def apply_filter(filter, arr):
    try:
        """
        This method convolves an image with a given kernel.
        Parameters
        ----------
        filter:array, Required
    
        arr: array, Required
    
        return tuple 
        """
    
        canvas = updated_arr(arr)
        img = updated_arr(arr)
        for i in range(1, len(arr) - 1):
            for j in range(1, len(arr[i]) - 1):
                pos = 0
                for rows in range(len(filter)):
                    for columns in range(len(filter[rows])):
                        x = rows - (len(filter) // 2)
                        y = columns - (len(filter[rows]) // 2)
                        pos += filter[rows][columns] * arr[i + x][j + y]
                canvas[i][j] = pos
    
                img[i][j] = 0 if pos < 0 else  255 if pos > 255 else  pos
        return (canvas, img)
    except:
        print("error in Apply_filter fn")

def corners_to_list(corners):
    try:
        """
        This methods change the list of corners into a list
    
        Parameters
        ----------
        corners: array, Required
    
        Return list of corners
        """
        corners_list = []
        for i in range(len(corners)):
            for j in range(len(corners[i])):
                if corners[i][j] > 0:
                    corners_list += [(i, j)]
        return corners_list
    except:
        print("Error in Corner_to_list function")

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

        
def hough_transform(img, corners, angle, radius, features):
    try:    
        """
        This method apply Hough Transform on a set of points.
        
        Parameters
        ----------
        img:array, Required
    
        corners:array, Required
    
        angle: int, Optional
        
        radius: Optional
        
        features: int, Optional
    
        return cordinates, list 
        """
        cordinates = []
        for i in range(features):
            cordinates += [[(0, 0), 0]]
        hough = []
        for i in range(radius):
            hough += [[]]
            for j in range(angle):
                hough[i] += [0]
        for i in range(len(corners)):
            for j in range(0, angle):
                r = corners[i][1]*math.cos(j * math.pi / angle) + corners[i][0]*math.sin(j * math.pi / angle)
                rbin = math.floor(radius/2 / math.sqrt(len(img)**2 + len(img[1])**2) * r + radius/2)
                r = math.floor(r)
                hough[rbin][j] += 1
                for k in range(features):
                    if hough[rbin][j] > cordinates[k][1]:
                        if [(r, j * math.pi / angle), hough[rbin][j]-1] in cordinates:
                            index = cordinates.index([(r, j * math.pi / angle), hough[rbin][j]-1])
                            cordinates = cordinates[:k] + [[(r, j * math.pi / angle), hough[rbin][j]]] + cordinates[k:index] + cordinates[index+1:]
                        else:
                            cordinates = cordinates[:k] + [[(r, j * math.pi / angle), hough[rbin][j]]] + cordinates[k:-1]
                        break
        return cordinates
    except:
        print("Error in hough_transform function")

def apply_hough(img, corners, angle = 180, radius = None, features = 4):
    try:
        """
        This method apply Hough Transofrm a specifed amount of times, and creates an image depicting the result.
        
        Parameters
        ----------
        img:array, Required
    
        corners:array, Required
    
        angle: int, Optional
        
        radius: Optional
        
        features: int, Optional
    
        return array 
    
        """
        if radius == None:
            radius = math.ceil(2*math.sqrt(len(img)**2 + len(img[1])**2))
        hough = hough_transform(img, corners, angle, radius, features)
        for i in range(features):
            pt1 = int(hough[i][0][0]/math.sin(hough[i][0][1]))
            pt2 = int((hough[i][0][0] - len(img[1])*math.cos(hough[i][0][1]))/math.sin(hough[i][0][1]))
            img = cv2.line(img, (0, pt1), (len(img[1]), pt2), (0, 0 , 255), 1)
        return img
    except:
        print("Error in apply_hough function")

if __name__ == "__main__":

    gaussian = [[0.077847, 0.123317, 0.077847], 
                [0.123317, 0.195346, 0.123317], 
                [0.077847, 0.123317, 0.077847]]
    horizontal_sobel = [[1, 2, 1], 
               [0, 0, 0], 
               [-1, -2, -1]]
    vertical_sobel = [[1, 0, -1], 
               [2, 0, -2], 
               [1, 0, -1]]
    
    img = cv2.imread("road.png", 0)
    arr = updated_arr(img)

    blurred = apply_filter(gaussian, arr)
    cv2.imwrite("gaussian_filter_image.png", blurred[1])

    h = apply_filter(horizontal_sobel, blurred[0])
    cv2.imwrite("sobel_filter_horizontal.png", h[1])
    v = apply_filter(vertical_sobel, blurred[0])
    cv2.imwrite("sobel_filter_vertical.png", v[1])

    edges = overlay_image(h[0], v[0])
    cv2.imwrite("edges_with_no_supression.png", edges[1])
    suppressed_edges = non_max_supresn(edges[0], h[0], v[0], "edges")
    cv2.imwrite("edges_with_supression.png", suppressed_edges[1])
    threshold_edges = threshold(suppressed_edges[0], 175, 60)
    cv2.imwrite("edges_threshold.png", threshold_edges)

    xxcord = apply_filter(horizontal_sobel, h[0])
    yycord = apply_filter(vertical_sobel, v[0])
    xycord = apply_filter(vertical_sobel, h[0])
    yxcord = apply_filter(horizontal_sobel, v[0])


    cv2.imwrite("xxcord.png", xxcord[1])
    cv2.imwrite("yycord.png", yycord[1])
    cv2.imwrite("xycord.png", xycord[1])
    cv2.imwrite("yxcord.png", yxcord[1])
    
    hess = hes_matrix(xxcord[0], yycord[0], xycord[0], yxcord[0], 175000)
    hess_threshold = non_max_supresn(hess[0], h[0], v[0], "corners")
    cv2.imwrite("corners.png", hess[1])
    cv2.imwrite("corners_threshold.png", hess_threshold[1])
    updated_hess = overlay_image(hess_threshold[0], threshold_edges / 4, bg=True) 
    cv2.imwrite("updated_corners.png", updated_hess[1])
    colored = (updated_hess[1]).copy()
    colored = cv2.cvtColor(updated_hess[1], cv2.COLOR_GRAY2RGB)
    ransac_image = colored.copy()
    ransac_image = apply_ransac(ransac_image, corners_to_list(hess_threshold[1]), it = 25)
    cv2.imwrite("ransac.png", ransac_image)
    HOUGH = colored.copy()
    HOUGH = apply_hough(HOUGH, corners_to_list(hess_threshold[1]), angle=45)
    cv2.imwrite("hough_image.png", HOUGH)
