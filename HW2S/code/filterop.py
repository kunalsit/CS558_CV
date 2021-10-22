import numpy as np
import math


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
    
        ar = updated_arr(arr)
        img = updated_arr(arr)
        for i in range(1, len(arr) - 1):
            for j in range(1, len(arr[i]) - 1):
                pos = 0
                for rows in range(len(filter)):
                    for columns in range(len(filter[rows])):
                        x = rows - (len(filter) // 2)
                        y = columns - (len(filter[rows]) // 2)
                        pos += filter[rows][columns] * arr[i + x][j + y]
                ar[i][j] = pos
    
                img[i][j] = 0 if pos < 0 else  255 if pos > 255 else  pos
        return (ar, img)
    except Exception as e:
        print("error in Apply_filter fn")
        print(e)


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
    except Exception as e:
        print("Error in overlay_image function")
        print(e)

def sup_nm(arr, h, v, mode):
    try:
        """
        This method supresses pixel intensities based on the neighborhood pixel, 
            
        Parameters
        ----------
        arr: array, Required
    
        h:array, Required
    
        v:array, Required
    
        mode : str , Required
    
        return list
        """
    
        ar = arr.copy()
        img = arr.copy()
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if mode == "edges":
                    angle = math.atan2(v[i][j], h[i][j]) 
                    ar[i][j] = arr[i][j]
                    if i == 0 or j == 0 or i == len(arr) - 1  or j == len(arr[i]) - 1:
                        ar[i][j] = 0
                    elif (angle >=  -1*math.pi/8 and angle <= math.pi / 8) or (angle > 7*math.pi/8 and angle <= -7*math.pi/8):
                        if arr[i][j] <= arr[i][j+1] or arr[i][j] <= arr[i][j-1]:
                            ar[i][j] = 0
                    elif (angle < -1*math.pi/8 and angle >= -3*math.pi/8) or (angle > math.pi/8 and angle <= 3*math.pi/8):
                        if arr[i][j] <= arr[i+1][j+1] or arr[i][j] <= arr[i-1][j-1]:
                            ar[i][j] = 0
                    elif (angle < -3*math.pi/8 and angle >= -5*math.pi/8) or (angle > 3*math.pi/8 and angle <= 5*math.pi/8):
                        if arr[i][j] <= arr[i+1][j] or arr[i][j] <= arr[i-1][j]:
                            ar[i][j] = 0
                    elif (angle < -5*math.pi/8 and angle >= -7*math.pi/8) or (angle > 5*math.pi/8 and angle <= 7*math.pi/8):
                        if arr[i][j] <= arr[i+1][j-1] or arr[i][j] <= arr[i-1][j+1]:
                            ar[i][j] = 0
                    else:
                        ar[i][j] = 0 
                    
                    
                    img[i][j] = 0 if ar[i][j] < 0 else 255 if ar[i][j] > 255 else ar[i][j]
                        
                elif mode == "corners":
                    ar[i][j] = arr[i][j]
                    if (i == 0 or j == 0 or i == len(arr) - 1 or j == len(arr[i]) - 1):
                        ar[i][j] = 0
                    elif not (arr[i][j] > arr[i+1][j+1] and arr[i][j] > arr[i-1][j-1] and arr[i][j] > arr[i+1][j-1] and arr[i][j] > arr[i-1][j+1] and arr[i][j] > arr[i][j+1] and arr[i][j] > arr[i][j-1] and arr[i][j] > arr[i+1][j] and arr[i][j] > arr[i-1][j]):
                        ar[i][j] = 0
    
                    img[i][j] = 0 if ar[i][j] < 0 else  255 if ar[i][j] > 255 else  ar[i][j]
                
                        
        return [ar, img]
    except Exception as e:
        print("Error in sup_nm fn")
        print(e)

def threshold(ar,min_value,max_value):
    try:
        
        """
        This method used to set pixel value to zero when it less than min_value,
        and if it is greater than min_value and less than max_value then set to 125 else 255.
            
        Parameters
        ----------
        ar: , Required
    
        min_value:int, Required
    
        max_value:int, Required
    
        return ar
        """
        canvas_len=len(ar)
        for i in range(canvas_len):
            ar_item_len=len(ar[i])
            for j in range(ar_item_len):
                ar[i][j] = 0 if ar[i][j] < min_value else  125 if ar[i][j] < max_value else 255
        return ar
    except Exception as e:
        print("Error in threshold function")
        print(e)


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
    except Exception as e:
        print("Error in hes_matrix fn")
        print(e)

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
    except Exception as e:
        print("Error updated_arr in Function")
        print(e)


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
    except Exception as e:
        print("Error in Corner_to_list function")
        print(e)
