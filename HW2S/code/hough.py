import math
import cv2

       
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