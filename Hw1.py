import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import math

img = None
kernel_radius = 0
root = tk.Tk()
root.title('HW1')
window_width = root.winfo_screenwidth()
window_height = root.winfo_screenheight()
width = 800
height = 600
left = int((window_width - width)/2)  
top = int((window_height - height)/2) 
root.geometry(f'{width}x{height}+{left}+{top}')
root.resizable(False,False)
rot = tk.StringVar() 
scale = tk.StringVar()
tx = tk.StringVar()
ty = tk.StringVar()

def load():
    file_path = filedialog.askopenfilename()
    global img
    img = cv2.imread(file_path) # read the image in BGR orders


def colorSeparation():
    b,g,r = cv2.split(img)
    zeros = np.zeros(img.shape[:2],dtype="uint8")
    b_img = cv2.merge([b,zeros,zeros])
    g_img = cv2.merge([zeros,g,zeros])
    r_img = cv2.merge([zeros,zeros,r])
    cv2.imshow("Blue channel",b_img)
    cv2.imshow("Green channel",g_img)
    cv2.imshow("Red channel",r_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def colorTransformation():
    b,g,r = cv2.split(img)
    cv_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    avg_gray = (b/3 + g/3 + r/3).astype(np.uint8)
    cv2.imshow("cv_gray",cv_gray)
    cv2.imshow("avg_gray",avg_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def colorExtraction():
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([18, 0, 25])
    upper_bound = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    mask_inverse = cv2.bitwise_not(mask)
    extracted_img = cv2.bitwise_and(img, img, mask=mask_inverse)
    cv2.imshow("Yellow-Green Mask",mask)
    cv2.imshow("Extracted image",extracted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def changeKernelRadius1(kernel_radius):
  blur = cv2.GaussianBlur(img, [2*kernel_radius+1, 2*kernel_radius+1], 0, 0)
  cv2.imshow('GaussianBlur', blur)
  
def changeKernelRadius2(kernel_radius):
  bilateral = cv2.bilateralFilter(img, 2*kernel_radius+1, 90, 90)
  cv2.imshow('BilateralFilter', bilateral)

def changeKernelRadius3(kernel_radius):
  median = cv2.medianBlur(img, 2*kernel_radius+1)
  cv2.imshow('MedianBlur', median)

def gaussianBlur():
  cv2.imshow('GaussianBlur', img)
  cv2.createTrackbar('m', 'GaussianBlur', 0, 5, changeKernelRadius1) 
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def bilateralFilter():
  cv2.imshow('BilateralFilter', img)
  cv2.createTrackbar('m', 'BilateralFilter', 0, 5, changeKernelRadius2) 
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def medianFilter():
  cv2.imshow('MedianBlur', img)
  cv2.createTrackbar('m', 'MedianBlur', 0, 5, changeKernelRadius3) 
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def sobelX():
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (3,3), 0, 0)
  sobelFilter = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
  vertical_edge = np.zeros([blur.shape[0], blur.shape[1]])
  for i in range(1,blur.shape[0]-1):
    for j in range(1, blur.shape[1]-1):
      value = (sobelFilter[0,0] * blur[i-1, j-1] +
              sobelFilter[0,1] * blur[i-1, j] +
              sobelFilter[0,2] * blur[i-1, j+1] +
              sobelFilter[1,0] * blur[i, j-1] +
              sobelFilter[1,1] * blur[i, j] +
              sobelFilter[1,2] * blur[i, j+1] +
              sobelFilter[2,0] * blur[i+1, j-1] +
              sobelFilter[2,1] * blur[i+1, j] +
              sobelFilter[2,2] * blur[i+1, j+1])
      vertical_edge[i, j] = value

  vertical_edge = np.clip(vertical_edge, 0, 255).astype(np.uint8) # set the pixel grayscale value < 0 to 0 and >255 to 255
  cv2.imshow('sobel x', vertical_edge)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def sobelY():
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (3,3), 0, 0)
  np.set_printoptions(threshold=np.inf)
  sobelFilter = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
  horizontal_edge = np.zeros([blur.shape[0], blur.shape[1]])
  for i in range(1,blur.shape[0]-1):
    for j in range(1, blur.shape[1]-1):
      value = (sobelFilter[0,0] * blur[i-1, j-1] +
              sobelFilter[0,1] * blur[i-1, j] +
              sobelFilter[0,2] * blur[i-1, j+1] +
              sobelFilter[1,0] * blur[i, j-1] +
              sobelFilter[1,1] * blur[i, j] +
              sobelFilter[1,2] * blur[i, j+1] +
              sobelFilter[2,0] * blur[i+1, j-1] +
              sobelFilter[2,1] * blur[i+1, j] +
              sobelFilter[2,2] * blur[i+1, j+1])
      horizontal_edge[i, j] = value

  horizontal_edge = np.clip(horizontal_edge, 0, 255).astype(np.uint8) # set the pixel grayscale value < 0 to 0 and >255 to 255
  cv2.imshow('sobel y', horizontal_edge)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def combination():
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (3,3), 0, 0)
  sobelFilterX = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
  vertical_edge = np.zeros([blur.shape[0], blur.shape[1]])
  for i in range(1,blur.shape[0]-1):
    for j in range(1, blur.shape[1]-1):
      value = (sobelFilterX[0,0] * blur[i-1, j-1] +
              sobelFilterX[0,1] * blur[i-1, j] +
              sobelFilterX[0,2] * blur[i-1, j+1] +
              sobelFilterX[1,0] * blur[i, j-1] +
              sobelFilterX[1,1] * blur[i, j] +
              sobelFilterX[1,2] * blur[i, j+1] +
              sobelFilterX[2,0] * blur[i+1, j-1] +
              sobelFilterX[2,1] * blur[i+1, j] +
              sobelFilterX[2,2] * blur[i+1, j+1])
      vertical_edge[i, j] = value  

  sobelFilterY = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
  horizontal_edge = np.zeros([blur.shape[0], blur.shape[1]])
  for i in range(1,blur.shape[0]-1):
    for j in range(1, blur.shape[1]-1):
      value = (sobelFilterY[0,0] * blur[i-1, j-1] +
              sobelFilterY[0,1] * blur[i-1, j] +
              sobelFilterY[0,2] * blur[i-1, j+1] +
              sobelFilterY[1,0] * blur[i, j-1] +
              sobelFilterY[1,1] * blur[i, j] +
              sobelFilterY[1,2] * blur[i, j+1] +
              sobelFilterY[2,0] * blur[i+1, j-1] +
              sobelFilterY[2,1] * blur[i+1, j] +
              sobelFilterY[2,2] * blur[i+1, j+1])
      horizontal_edge[i, j] = value

  combine_image = np.zeros([blur.shape[0], blur.shape[1]])
  for i in range(blur.shape[0]):
    for j in range(blur.shape[1]):
      new_pixel_value = np.sqrt(vertical_edge[i, j]**2 + horizontal_edge[i, j]**2)
      combine_image[i, j] = new_pixel_value

  combine_image = np.clip(combine_image, 0, 255).astype(np.uint8)
  normalized = cv2.normalize(combine_image, None, 0, 255, cv2.NORM_MINMAX)
  _, result1 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
  _, result2 = cv2.threshold(normalized, 28, 255, cv2.THRESH_BINARY)
  cv2.imshow('sobel xy', normalized)
  cv2.imshow('threshold_1', result1)
  cv2.imshow('threshold_2', result2)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def gradientAngle():
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (3,3), 0, 0)
  sobelFilterX = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
  vertical_edge = np.zeros([blur.shape[0], blur.shape[1]])
  for i in range(1,blur.shape[0]-1):
    for j in range(1, blur.shape[1]-1):
      value = (sobelFilterX[0,0] * blur[i-1, j-1] +
              sobelFilterX[0,1] * blur[i-1, j] +
              sobelFilterX[0,2] * blur[i-1, j+1] +
              sobelFilterX[1,0] * blur[i, j-1] +
              sobelFilterX[1,1] * blur[i, j] +
              sobelFilterX[1,2] * blur[i, j+1] +
              sobelFilterX[2,0] * blur[i+1, j-1] +
              sobelFilterX[2,1] * blur[i+1, j] +
              sobelFilterX[2,2] * blur[i+1, j+1])
      vertical_edge[i, j] = value

  sobelFilterY = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
  horizontal_edge = np.zeros([blur.shape[0], blur.shape[1]])
  for i in range(1,blur.shape[0]-1):
    for j in range(1, blur.shape[1]-1):
      value = (sobelFilterY[0,0] * blur[i-1, j-1] +
              sobelFilterY[0,1] * blur[i-1, j] +
              sobelFilterY[0,2] * blur[i-1, j+1] +
              sobelFilterY[1,0] * blur[i, j-1] +
              sobelFilterY[1,1] * blur[i, j] +
              sobelFilterY[1,2] * blur[i, j+1] +
              sobelFilterY[2,0] * blur[i+1, j-1] +
              sobelFilterY[2,1] * blur[i+1, j] +
              sobelFilterY[2,2] * blur[i+1, j+1])
      horizontal_edge[i, j] = value

  combine_image = np.zeros([blur.shape[0], blur.shape[1]])
  for i in range(blur.shape[0]):
    for j in range(blur.shape[1]):
      new_pixel_value = np.sqrt(vertical_edge[i, j]**2 + horizontal_edge[i, j]**2)
      combine_image[i, j] = new_pixel_value

  combine_image = np.clip(combine_image, 0, 255).astype(np.uint8)
  angle = np.arctan2(horizontal_edge, vertical_edge) * (180 / np.pi)
  angle = (angle+360) % 360 # angle [0, 360]
  mask1 = np.zeros_like(combine_image)
  mask2 = np.zeros_like(combine_image)
  for i in range(combine_image.shape[0]):
    for j in range(combine_image.shape[1]):
      if angle[i, j] <= 190 and angle[i, j] >= 170:
        mask1[i, j] = 255
      if angle[i, j] <= 280 and angle[i, j] >= 260:
        mask2[i, j] = 255
  output1 = cv2.bitwise_and(combine_image, mask1)
  output2 = cv2.bitwise_and(combine_image, mask2)
  cv2.imshow('angle1', output1)
  cv2.imshow('angle2', output2)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def transforms():
  try:
    angle = float(entry1.get())
    scale = float(entry2.get())
    x = float(entry3.get())
    y = float(entry4.get())
  except:
    angle = float(0)
    scale = float(1)
    x = float(0)
    y = float(0)

  M_translation = np.array([[1, 0, x],
                            [0, 1, y],
                            [0, 0, 1]])

  M_rotate = cv2.getRotationMatrix2D((240,200), angle, scale)
  M_rotate = np.insert(M_rotate, 2, values=[0,0,1], axis=0)

  M = np.matmul(M_translation, M_rotate)
  M = np.delete(M, -1 ,axis=0)

  result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

  cv2.namedWindow('Transformed Image', cv2.WINDOW_NORMAL)
  cv2.imshow('Transformed Image', result)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

label1 = tk.Label(root,
                  text='1. Image Processing',
                  font=('Arial',10,'bold'),
                )

label2 = tk.Label(root,
                  text='2. Image Smoothing',
                  font=('Arial',10,'bold'),
                )

label3 = tk.Label(root,
                  text='3. Edge Detection',
                  font=('Arial',10,'bold'),
                )

label4 = tk.Label(root,
                  text='4. Transforms',
                  font=('Arial',10,'bold'),
                )

label5 = tk.Label(root,
                  text='Rotation:',
                  font=('Arial',10,'bold'),
                )

label6 = tk.Label(root,
                  text='Scaling:',
                  font=('Arial',10,'bold'),
                )

label7 = tk.Label(root,
                  text='Tx:',
                  font=('Arial',10,'bold'),
                )
          
label8 = tk.Label(root,
                  text='Ty',
                  font=('Arial',10,'bold'),
                )

label9 = tk.Label(root,
                  text='deg',
                  font=('Arial',10,'bold'),
                )

label10 = tk.Label(root,
                  text='pixel',
                  font=('Arial',10,'bold'),
                )

label11 = tk.Label(root,
                  text='pixel',
                  font=('Arial',10,'bold'),
                )              
entry1 = tk.Entry(root,
                  justify='left',
                  textvariable=rot,
              )

entry2 = tk.Entry(root,
                  justify='left',
                  textvariable=scale,
              )

entry3 = tk.Entry(root,
                  justify='left',
                  textvariable=tx,
              )

entry4 = tk.Entry(root,
                  justify='left',
                  textvariable=ty,
              )
btn1 = tk.Button(root,
                text='Load Image 1',
                font=('Arial',10,'bold'),
                command=load
              )
btn2 = tk.Button(root,
                text='Load Image 2',
                font=('Arial',10,'bold'),
                command=load
              )
btn1_1 = tk.Button(root,
                text='1.1 Color Separation',
                font=('Arial',10,'bold'),
                command=colorSeparation
              )
btn1_2 = tk.Button(root,
                text='1.2 Color Transformation',
                font=('Arial',10,'bold'),
                command=colorTransformation
              )
btn1_3 = tk.Button(root,
                text='1.3 Color Extraction',
                font=('Arial',10,'bold'),
                command=colorExtraction
              )

btn2_1 = tk.Button(root,
                text='2.1 Gaussian Blur',
                font=('Arial',10,'bold'),
                command=gaussianBlur
              )

btn2_2 = tk.Button(root,
                text='2.2 Bilateral Filter',
                font=('Arial',10,'bold'),
                command=bilateralFilter
              )

btn2_3 = tk.Button(root,
                text='2.3 Median Filter',
                font=('Arial',10,'bold'),
                command=medianFilter
              )

btn3_1 = tk.Button(root,
                text='3.1 Sobel X',
                font=('Arial',10,'bold'),
                command=sobelX
              )

btn3_2 = tk.Button(root,
                text='3.2 Sobel Y',
                font=('Arial',10,'bold'),
                command=sobelY
              )

btn3_3 = tk.Button(root,
                text='3.3 Combination and Threshold',
                font=('Arial',10,'bold'),
                command=combination
              )

btn3_4 = tk.Button(root,
                text='3.4 Gradient Angle',
                font=('Arial',10,'bold'),
                command=gradientAngle
              )

btn4 = tk.Button(root,
                text='4. Transforms',
                font=('Arial',10,'bold'),
                command=transforms
              )

btn1.place(x=50,y=230)
btn2.place(x=50,y=330)

label1.place(x=200,y=20)
btn1_1.place(x=220,y=50)
btn1_2.place(x=220,y=100)
btn1_3.place(x=220,y=150)

label2.place(x=200,y=200)
btn2_1.place(x=220,y=230)
btn2_2.place(x=220,y=280)
btn2_3.place(x=220,y=330)

label3.place(x=200,y=380)
btn3_1.place(x=220,y=410)
btn3_2.place(x=220,y=460)
btn3_3.place(x=220,y=510)
btn3_4.place(x=220,y=560)

label4.place(x=500,y=20)
label5.place(x=500,y=60)
label6.place(x=500,y=100)
label7.place(x=500,y=140)
label8.place(x=500,y=180)
entry1.place(x=570,y=63)
entry2.place(x=570,y=103)
entry3.place(x=570,y=143)
entry4.place(x=570,y=183)
label9.place(x=730,y=60)
label10.place(x=730,y=140)
label11.place(x=730,y=180)
btn4.place(x=590, y=220)

root.mainloop()