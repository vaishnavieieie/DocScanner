# %%
import cv2
# from google.colab.patches import cv2_imshow
import numpy as np

# %%
img_width=480
img_height=640

# %%
image=cv2.imread('D:\BTI_sem9\IP\project\images\4.jpg')

# %%
def show_image(image):
  # cv2_imshow(image)
  cv2.imshow('image',image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# %%
image=cv2.resize(image,(img_width,img_height))
show_image(image)

# %%
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_blur=cv2.GaussianBlur(image_gray,(5,5),1)

# %%
image_canny=cv2.Canny(image_blur,100,200)
kernel=np.ones((3,3))

# %%
image_closed=cv2.erode(cv2.dilate(image_canny,kernel,iterations=2),kernel, iterations=1)


# %%
show_image(image_closed)

# %%
# finding all contours
image_contours=image.copy()
image_f_contours=image.copy()
contours,hierarchy=cv2.findContours(image_closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_contours,contours,-1,(0,255,0),10)

# %%
show_image(image_contours)

# %%
# finding biggest contours based on a threshold
def find_biggest_contour(contours):
  contours=sorted(contours,key=cv2.contourArea,reverse=True)
  biggest_contour=np.array([])
  max_area=0
  for i in contours:
    area=cv2.contourArea(i)
    if area>4000:
      perimeter=cv2.arcLength(i,True)
      # to get the contour approximated as a rectangle
      app=cv2.approxPolyDP(i,0.02*perimeter,True)
      if area>max_area and len(app)==4:
        biggest_contour=app
        max_area=area
  return biggest_contour,max_area

# %%
biggest_contour,max_area=find_biggest_contour(contours)
biggest_contour

# %%
cv2.drawContours(image_f_contours,biggest_contour,-1,(0,255,0),10)
show_image(image_f_contours)

# %%

if biggest_contour.size!=0:
  final_points=np.float32([[img_width,0],[0,0],[0 ,img_height],[img_width,img_height]])
  perspective=cv2.getPerspectiveTransform(biggest_contour.astype(np.float32),final_points)
  warped_image=cv2.warpPerspective(image,perspective,(img_width,img_height))


# %%
show_image(warped_image)

# %%
# make pdf

# %%
# make UI


