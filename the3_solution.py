# Andaç Berkay Seval 2235521
# Asrın Doğrusöz 2380301

import cv2
import os
import numpy as np
from skimage import io, img_as_ubyte, color
from skimage import exposure

INPUT_PATH = "./THE3_Images/"
OUTPUT_PATH = "./Outputs/"                    

def detect_faces(img, n):
    ### for image 1:
    if n == 1:
        ### apply histogram equalization to image
        img = exposure.equalize_hist(img)
        img = img_as_ubyte(img)
        ### convert image from RGB to BGR for opencv manipulations
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ### convert image from RGB color space to YCbCr color space
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite(OUTPUT_PATH + "ycrcb_1.png", YCrCb) 
        ### for image dimensions, if the color of the pixel is not in the cluster, paint it black in RGB space
        for i in range(356):
            for j in range(420):
                if YCrCb[i,j,1] < 153 or YCrCb[i,j,1] > 170:
                    YCrCb[i,j] = [16,128,128]
                if YCrCb[i,j,2] < 100 or YCrCb[i,j,2] > 150:
                    YCrCb[i,j] = [16,128,128]
        ### convert image back to RGB color space for masked image
        img_bgr = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
        # cv2.imwrite(OUTPUT_PATH + "masked_1.png", img_bgr)
        ### convert image to gray scale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(OUTPUT_PATH + "gray_1.png", gray)
        ### convert gray scale image to binary image
        ret, binary = cv2.threshold(gray,44,255,cv2.THRESH_BINARY)
        # cv2.imwrite(OUTPUT_PATH + "binary_1.png", binary)
        ### find contours of the binary image to draw red rectangles around the faces in the original image
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rectangle = cv2.boundingRect(c)
            if rectangle[2] < 55 and rectangle[3] < 80: continue
            x, y, w, h = rectangle
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imwrite(OUTPUT_PATH + "1_faces.png", image)

    ### for image 2:
    elif n == 2:
        ### convert image from RGB color space to YCbCr color space
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite(OUTPUT_PATH + "ycrcb_2.png", YCrCb) 
        ### for image dimensions, if the color of the pixel is not in the cluster, paint it black in RGB space
        for i in range(4032):
            for j in range(3024):
                if YCrCb[i,j,1] < 152 or YCrCb[i,j,1] > 190:
                    YCrCb[i,j] = [16,128,128]
                if YCrCb[i,j,2] < 100 or YCrCb[i,j,2] > 150:
                    YCrCb[i,j] = [16,128,128]
        ### convert image back to RGB color space for masked image
        img_bgr = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
        # cv2.imwrite(OUTPUT_PATH + "masked_2.png", img_bgr)
        ### convert image to gray scale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(OUTPUT_PATH + "gray_2.png", gray)
        ### convert gray scale image to binary image
        ret, binary = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
        # cv2.imwrite(OUTPUT_PATH + "binary_2.png", binary)
        ### find contours of the binary image to draw red rectangles around the faces in the original image
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rectangle = cv2.boundingRect(c)
            if rectangle[2] < 300 or rectangle[3] < 300: continue
            x, y, w, h = rectangle
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imwrite(OUTPUT_PATH + "2_faces.png", image)

    ### for image 3:
    else:
        ### convert image from RGB color space to YCbCr color space
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite(OUTPUT_PATH + "ycrcb_3.png", YCrCb)
        ### for image dimensions, if the color of the pixel is not in the cluster, paint it black in RGB space
        for i in range(250):
            for j in range(200):
                if YCrCb[i,j,1] < 152 or YCrCb[i,j,1] > 190:
                    YCrCb[i,j] = [16,128,128]
                if YCrCb[i,j,2] < 100 or YCrCb[i,j,2] > 150:
                    YCrCb[i,j] = [16,128,128]
        ### convert image back to RGB color space for masked image
        img_bgr = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
        # cv2.imwrite(OUTPUT_PATH + "masked_3.png", img_bgr)
        ### convert image to gray scale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(OUTPUT_PATH + "gray_3.png", gray)
        ### convert gray scale image to binary image
        ret, binary = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
        # cv2.imwrite(OUTPUT_PATH + "binary_3.png", binary)
        ### find contours of the binary image to draw red rectangles around the faces in the original image
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rectangle = cv2.boundingRect(c)
            if rectangle[2] < 20 or rectangle[3] < 20: continue
            x, y, w, h = rectangle
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imwrite(OUTPUT_PATH + "3_faces.png", image)

def color_images(n):
    ### for image 1:
    if n == 1:
        img = io.imread(INPUT_PATH + "1.png")
        l = []
        for i in range(3024):
            for j in range(4032):
                if img[i, j] not in l:
                    l.append(img[i, j])
        ### 239 different gray values
        l.sort()
        ### cluster gray values for color mapping
        l1 = l[:24]
        l2 = l[24:48]
        l3 = l[48:72]
        l4 = l[72:96]
        l5 = l[96:120]
        l6 = l[120:144]
        l7 = l[144:168]
        l8 = l[168:192]
        l9 = l[192:216]
        l10 = l[216:]

        ### apply k-means clustering to source image to cluster colors
        img2 = cv2.imread(INPUT_PATH + "1_source.png")
        img2 = img2.reshape((-1,3))
        img2 = np.float32(img2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 10
        ret, label, center=cv2.kmeans(img2,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        # res = center[label.flatten()]
        # result = res.reshape((356,420,3))
        # cv2.imwrite(OUTPUT_PATH + "kmeans_1.png", result)
        ### output colored image
        out = np.zeros((3024,4032,3), dtype=np.uint8)
        ### based on gray values of original image, map the center colors of source image obtained from
        ### k-means clustering
        for i in range(3024):
            for j in range(4032):
                if img[i,j] in l1:
                    out[i,j] = center[0]
                elif img[i,j] in l2:
                    out[i,j] = center[1]
                elif img[i,j] in l3:
                    out[i,j] = center[2]
                elif img[i,j] in l4:
                    out[i,j] = center[3]
                elif img[i,j] in l5:
                    out[i,j] = center[4]
                elif img[i,j] in l6:
                    out[i,j] = center[5]
                elif img[i,j] in l7:
                    out[i,j] = center[6]
                elif img[i,j] in l8:
                    out[i,j] = center[7]
                elif img[i,j] in l9:
                    out[i,j] = center[8]
                else:
                    out[i,j] = center[9]
        cv2.imwrite(OUTPUT_PATH + "1_colored.png", out)
        # cv2.imwrite(OUTPUT_PATH + "blue_1.png", out[:,:,0])
        # cv2.imwrite(OUTPUT_PATH + "green_1.png", out[:,:,1])
        # cv2.imwrite(OUTPUT_PATH + "red_1.png", out[:,:,2])
        # out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        # out_hsi = color.rgb2hsv(out)
        # io.imsave(OUTPUT_PATH + "hue_1.png", out_hsi[:,:,0])
        # io.imsave(OUTPUT_PATH + "saturation_1.png", out_hsi[:,:,1])
        # io.imsave(OUTPUT_PATH + "intensity_1.png", out_hsi[:,:,2])

    ### for image 2:
    elif n == 2:
        img = io.imread(INPUT_PATH + "2.png")
        l = []
        for i in range(2643):
            for j in range(3904):
                if img[i, j] not in l:
                    l.append(img[i, j])
        ### 255 different gray values 
        l.sort()
        ### cluster gray values for color mapping
        l1 = l[:25]
        l2 = l[25:50]
        l3 = l[50:75]
        l4 = l[75:100]
        l5 = l[100:125]
        l6 = l[125:150]
        l7 = l[150:175]
        l8 = l[175:200]
        l9 = l[200:225]
        l10 = l[225:]  

        ### apply k-means clustering to source image to cluster colors
        img2 = cv2.imread(INPUT_PATH + "2_source.png")
        img2 = img2.reshape((-1,3))
        img2 = np.float32(img2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 10
        ret, label, center=cv2.kmeans(img2,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        # res = center[label.flatten()]
        # result = res.reshape((4032,3024,3))
        # cv2.imwrite(OUTPUT_PATH + "kmeans_2.png", result)
        ### output colored image
        out = np.zeros((2643,3904,3), dtype=np.uint8)
        ### based on gray values of original image, map the center colors of source image obtained from
        ### k-means clustering
        for i in range(2643):
            for j in range(3904):
                if img[i,j] in l1:
                    out[i,j] = center[0]
                elif img[i,j] in l2:
                    out[i,j] = center[1]
                elif img[i,j] in l3:
                    out[i,j] = center[2]
                elif img[i,j] in l4:
                    out[i,j] = center[3]
                elif img[i,j] in l5:
                    out[i,j] = center[4]
                elif img[i,j] in l6:
                    out[i,j] = center[5]
                elif img[i,j] in l7:
                    out[i,j] = center[6]
                elif img[i,j] in l8:
                    out[i,j] = center[7]
                elif img[i,j] in l9:
                    out[i,j] = center[8]
                else:
                    out[i,j] = center[9]
        cv2.imwrite(OUTPUT_PATH + "2_colored.png", out)  
        # cv2.imwrite(OUTPUT_PATH + "blue_2.png", out[:,:,0])
        # cv2.imwrite(OUTPUT_PATH + "green_2.png", out[:,:,1])
        # cv2.imwrite(OUTPUT_PATH + "red_2.png", out[:,:,2])
        # out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        # out_hsi = color.rgb2hsv(out)
        # io.imsave(OUTPUT_PATH + "hue_2.png", out_hsi[:,:,0])
        # io.imsave(OUTPUT_PATH + "saturation_2.png", out_hsi[:,:,1])
        # io.imsave(OUTPUT_PATH + "intensity_2.png", out_hsi[:,:,2])
    ### for image 3:
    elif n == 3:
        img = io.imread(INPUT_PATH + "3.png")
        l = []
        for i in range(4032):
            for j in range(2784):
                if img[i, j] not in l:
                    l.append(img[i, j])
        ### 256 different gray values 
        l.sort()
        ### cluster gray values for color mapping
        l1 = l[:25]
        l2 = l[25:50]
        l3 = l[50:75]
        l4 = l[75:100]
        l5 = l[100:125]
        l6 = l[125:150]
        l7 = l[150:175]
        l8 = l[175:200]
        l9 = l[200:225]
        l10 = l[225:]   

        ### apply k-means clustering to source image to cluster colors
        img2 = cv2.imread(INPUT_PATH + "3_source.png")
        img2 = img2.reshape((-1,3))
        img2 = np.float32(img2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 16
        ret, label, center=cv2.kmeans(img2,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        # res = center[label.flatten()]
        # result = res.reshape((250,200,3))
        # cv2.imwrite(OUTPUT_PATH + "kmeans_3.png", result)
        ### output colored image
        out = np.zeros((4032,2784,3), dtype=np.uint8)
        ### based on gray values of original image, map the center colors of source image obtained from
        ### k-means clustering
        for i in range(4032):
            for j in range(2784):
                if img[i,j] in l1:
                    out[i,j] = center[0]
                elif img[i,j] in l2:
                    out[i,j] = center[1]
                elif img[i,j] in l3:
                    out[i,j] = center[2]
                elif img[i,j] in l4:
                    out[i,j] = center[3]
                elif img[i,j] in l5:
                    out[i,j] = center[4]
                elif img[i,j] in l6:
                    out[i,j] = center[5]
                elif img[i,j] in l7:
                    out[i,j] = center[6]
                elif img[i,j] in l8:
                    out[i,j] = center[7]
                elif img[i,j] in l9:
                    out[i,j] = center[8]
                else:
                    out[i,j] = center[9]
        cv2.imwrite(OUTPUT_PATH + "3_colored.png", out) 
        # cv2.imwrite(OUTPUT_PATH + "blue_3.png", out[:,:,0])
        # cv2.imwrite(OUTPUT_PATH + "green_3.png", out[:,:,1])
        # cv2.imwrite(OUTPUT_PATH + "red_3.png", out[:,:,2])
        # out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        # out_hsi = color.rgb2hsv(out)
        # io.imsave(OUTPUT_PATH + "hue_3.png", out_hsi[:,:,0])
        # io.imsave(OUTPUT_PATH + "saturation_3.png", out_hsi[:,:,1])
        # io.imsave(OUTPUT_PATH + "intensity_3.png", out_hsi[:,:,2])     

    ### for image 4:
    else:
        img = io.imread(INPUT_PATH + "4.png")
        img2 = io.imread(INPUT_PATH + "4_source.png")
        ### 5 different gray values
        l = []
        l2 = []
        ### output colored image
        out = np.zeros((600,900,3), dtype=np.uint8)
        for i in range(600):
            for j in range(900):
                if img[i,j] not in l:
                    l.append(img[i,j])
                if [img2[i,j,0], img2[i,j,1], img2[i,j,2]] not in l2:
                    l2.append([img2[i,j,0], img2[i,j,1], img2[i,j,2]])
        ### direct color mapping since number of different colors are equal
        for i in range(600):
            for j in range(900):
                if img[i,j] == l[0]:
                    out[i,j] = l2[0]
                elif img[i,j] == l[1]:
                    out[i,j] = l2[1]
                elif img[i,j] == l[2]:
                    out[i,j] = l2[2]
                elif img[i,j] == l[3]:
                    out[i,j] = l2[3]
                else:
                    out[i,j] = l2[4]
        io.imsave(OUTPUT_PATH + "4_colored.png", out)
        # io.imsave(OUTPUT_PATH + "blue_4.png", out[:,:,2])
        # io.imsave(OUTPUT_PATH + "green_4.png", out[:,:,1])
        # io.imsave(OUTPUT_PATH + "red_4.png", out[:,:,0])
        # out_hsi = color.rgb2hsv(out)
        # io.imsave(OUTPUT_PATH + "hue_4.png", out_hsi[:,:,0])
        # io.imsave(OUTPUT_PATH + "saturation_4.png", out_hsi[:,:,1])
        # io.imsave(OUTPUT_PATH + "intensity_4.png", out_hsi[:,:,2])


def detect_edges(img, n):
    ### sobel filters (gradient filters) for vertical and horizontal edges
    sobelV = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype = "int")

    sobelH = np.array((
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]), dtype = "int")

    ### for image 1:
    if n == 1:
        r, g, b = cv2.split(img)
        r = cv2.filter2D(r, -1, sobelV)
        g = cv2.filter2D(g, -1, sobelV)
        b = cv2.filter2D(b, -1, sobelV)
        out = cv2.merge((r, g, b))
        io.imsave(OUTPUT_PATH + "1_rgb_vertical_colored_edges.png", out)

        r, g, b = cv2.split(img)
        r = cv2.filter2D(r, -1, sobelH)
        g = cv2.filter2D(g, -1, sobelH)
        b = cv2.filter2D(b, -1, sobelH)
        out = cv2.merge((r, g, b))
        io.imsave(OUTPUT_PATH + "1_rgb_horizontal_colored_edges.png", out)

        img = color.rgb2hsv(img)
        h, s, v = cv2.split(img)
        h = cv2.filter2D(h, -1, sobelV)
        s = cv2.filter2D(s, -1, sobelV)
        v = cv2.filter2D(v, -1, sobelV)
        out = cv2.merge((h, s, v))
        io.imsave(OUTPUT_PATH + "1_hsv_vertical_colored_edges.png", out)

        img = color.rgb2hsv(img)
        h, s, v = cv2.split(img)
        h = cv2.filter2D(h, -1, sobelH)
        s = cv2.filter2D(s, -1, sobelH)
        v = cv2.filter2D(v, -1, sobelH)
        out = cv2.merge((h, s, v))
        io.imsave(OUTPUT_PATH + "1_hsv_horizontal_colored_edges.png", out)

    ### for image 2:
    elif n == 2:
        r, g, b = cv2.split(img)
        r = cv2.filter2D(r, -1, sobelV)
        g = cv2.filter2D(g, -1, sobelV)
        b = cv2.filter2D(b, -1, sobelV)
        out = cv2.merge((r, g, b))
        io.imsave(OUTPUT_PATH + "2_rgb_vertical_colored_edges.png", out)

        r, g, b = cv2.split(img)
        r = cv2.filter2D(r, -1, sobelH)
        g = cv2.filter2D(g, -1, sobelH)
        b = cv2.filter2D(b, -1, sobelH)
        out = cv2.merge((r, g, b))
        io.imsave(OUTPUT_PATH + "2_rgb_horizontal_colored_edges.png", out)

        img = color.rgb2hsv(img)
        h, s, v = cv2.split(img)
        h = cv2.filter2D(h, -1, sobelV)
        s = cv2.filter2D(s, -1, sobelV)
        v = cv2.filter2D(v, -1, sobelV)
        out = cv2.merge((h, s, v))
        io.imsave(OUTPUT_PATH + "2_hsv_vertical_colored_edges.png", out)

        img = color.rgb2hsv(img)
        h, s, v = cv2.split(img)
        h = cv2.filter2D(h, -1, sobelH)
        s = cv2.filter2D(s, -1, sobelH)
        v = cv2.filter2D(v, -1, sobelH)
        out = cv2.merge((h, s, v))
        io.imsave(OUTPUT_PATH + "2_hsv_horizontal_colored_edges.png", out)  

    ### for image 3:
    else:
        r, g, b = cv2.split(img)
        r = cv2.filter2D(r, -1, sobelV)
        g = cv2.filter2D(g, -1, sobelV)
        b = cv2.filter2D(b, -1, sobelV)
        out = cv2.merge((r, g, b))
        io.imsave(OUTPUT_PATH + "3_rgb_vertical_colored_edges.png", out)

        r, g, b = cv2.split(img)
        r = cv2.filter2D(r, -1, sobelH)
        g = cv2.filter2D(g, -1, sobelH)
        b = cv2.filter2D(b, -1, sobelH)
        out = cv2.merge((r, g, b))
        io.imsave(OUTPUT_PATH + "3_rgb_horizontal_colored_edges.png", out)

        img = color.rgb2hsv(img)
        h, s, v = cv2.split(img)
        h = cv2.filter2D(h, -1, sobelV)
        s = cv2.filter2D(s, -1, sobelV)
        v = cv2.filter2D(v, -1, sobelV)
        out = cv2.merge((h, s, v))
        io.imsave(OUTPUT_PATH + "3_hsv_vertical_colored_edges.png", out)

        img = color.rgb2hsv(img)
        h, s, v = cv2.split(img)
        h = cv2.filter2D(h, -1, sobelH)
        s = cv2.filter2D(s, -1, sobelH)
        v = cv2.filter2D(v, -1, sobelH)
        out = cv2.merge((h, s, v))
        io.imsave(OUTPUT_PATH + "3_hsv_horizontal_colored_edges.png", out)  

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    ### when reading image 1 with skimage, it has a dimension of (356, 420, 4), the image[:,:,3] array is all 255.
    ### thus, I delete that array since it has no purpose.
    img1 = io.imread(INPUT_PATH + "1_source.png")
    img1 = img1.reshape((356*420, 4))
    img1 = np.delete(img1, 3, 1)
    img1 = img1.reshape((356, 420, 3))

    detect_faces(img1, 1)
    detect_edges(img1, 1)

    img2 = cv2.imread(INPUT_PATH + "2_source.png")
    detect_faces(img2, 2)
    img2 = io.imread(INPUT_PATH + "2_source.png")
    detect_edges(img2, 2)

    img3 = cv2.imread(INPUT_PATH + "3_source.png")
    detect_faces(img3, 3)
    img3 = io.imread(INPUT_PATH + "3_source.png")
    detect_edges(img3, 3)   

    color_images(1)
    color_images(2)
    color_images(3)
    color_images(4)

