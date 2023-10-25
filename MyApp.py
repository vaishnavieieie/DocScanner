import streamlit as st
from PIL import Image
import io
import cv2
import numpy as np
import img2pdf
import base64
import os

# image array
modified_images=[]

# font
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
color = (0, 0, 0)
thickness = 2
text=''
coverImage=None
img_width=480
img_height=640


def find_biggest_contour(contours):
  contours=sorted(contours,key=cv2.contourArea,reverse=True)
  biggest_contour=np.array([])
  max_area=0
  for i in contours:
    area=cv2.contourArea(i)
    if area>4000:
      perimeter=cv2.arcLength(i,True)

      app=cv2.approxPolyDP(i,0.02*perimeter,True)
      if area>max_area and len(app)==4:
        biggest_contour=app
        max_area=area
  return biggest_contour,max_area

def process_image(image):


    image=cv2.resize(image,(img_width,img_height))
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_blur=cv2.GaussianBlur(image_gray,(5,5),1)
    image_canny=cv2.Canny(image_blur,100,200)
    kernel=np.ones((3,3))
    image_closed=cv2.erode(cv2.dilate(image_canny,kernel,iterations=2),kernel, iterations=1)
    contours,hierarchy=cv2.findContours(image_closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour,max_area=find_biggest_contour(contours)
    warped_image=image.copy()
    # path_name="dummy.pdf"
    # pil_images=None
    if biggest_contour.size!=0:
        final_points=np.float32([[img_width,0],[0,0],[0 ,img_height],[img_width,img_height]])
        perspective=cv2.getPerspectiveTransform(biggest_contour.astype(np.float32),final_points)
        warped_image=cv2.warpPerspective(image,perspective,(img_width,img_height))
    return warped_image

def add_cover(text):
   image_white=cv2.imread("cover.jpg")
   image_white=cv2.resize(image_white,(480,640))
   text_size = cv2.getTextSize(text, font, fontScale, thickness)[0]
   max_width = min(text_size[0], img_width)
   max_height = min(text_size[1], img_height)
   x=(img_width-max_width)//2
   y=(img_height-max_height)//2
   x=max(x,0)
   y=max(y,0)
#    todo add text to next line if it exceeds the image width
   cv2.putText(image_white,text,(x,y),font,fontScale,color,thickness,cv2.LINE_AA)
   return image_white


 
st.title("Document Scanner")
st.write("Get scanned document in 3 easy steps:")
st.write("1. Upload the image/images")
st.write("2. Select cover options")
st.write("3. Click on Process Image and download the pdf!")

images = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"],accept_multiple_files=True)
agree = st.checkbox('Add a cover page?')
if agree:
    text = st.text_input('Cover page title', 'Image processing Assignment')
    st.write('Title page will display: ', text)
    coverImage=add_cover(text)



if images is not None:
    for image in images:

        st.image(image, use_column_width=True)
        
    if st.button("Process Image"):

        for image in images:
            image_data = image.read()
            img = Image.open(io.BytesIO(image_data))
            processed_img = process_image(np.array(img))
            modified_images.append(processed_img)

        path_name="dummy.pdf"
        buff = io.BytesIO()
        modified_images=map(lambda x:Image.fromarray(x),modified_images)
        images_mod=[]
        if agree:
            images_mod.append(Image.fromarray(coverImage))
        for i in modified_images:
            images_mod.append(i)

        images_mod[0].save(path_name,save_all=True,append_images=images_mod[1:])
        with open("dummy.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label="Export_Report",
                data=PDFbyte,
                file_name="test.pdf",
                mime='application/octet-stream')
        

else:
    st.write("Please upload an image.")
