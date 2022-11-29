import cv2
import numpy as np
import math
from collections import defaultdict
import streamlit as st
from PIL import Image
import random


def add_noise(img):
    # Getting the dimensions of the image
    row, col,channel = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


#Blurring an Image using a Custom 2D-Convolution Kernel "Apply blurring kernel"

def Custom_Blur(img,k_size):
    blur_kernel = np.ones((k_size, k_size), np.float32) / 100
    blur_image = cv2.filter2D(src=img, ddepth=-1, kernel=blur_kernel)
    return blur_image

#  Blurring an Image using built-in Blur Function

def Blur_function(img,k_size):
    blur_image = cv2.blur(src=img, ksize=k_size)
    return blur_image

# Blurring an Image using Median Blur Method
def Median_Blur(img,k_size):
    blur_image = cv2.medianBlur(src=img, ksize=k_size)
    return blur_image

# Blurring of Image using Gaussian Blur Method
def Gaussian_Blur(img,k_size):
    blur_image = cv2.GaussianBlur(src=img, ksize=k_size, sigmaX=1, sigmaY=1)
    return blur_image

def main_opration():
    st.title("Compuetr Vision")
    st.header("Practical No. 01")

    # st.subheader("")
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    if st.checkbox("Add Sand Sand and Peppaer noise to image", key="disabled"):
        processed_image = add_noise(original_image)
    else:
        processed_image = original_image

    filter = st.radio(
        "➳ Choose your Favourite Filter 👇",
        ["Custom 2D Filter","Built-in Blur","Median Blur","Gaussian Blur "],
        key="filter",
    )

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
             unsafe_allow_html=True)

    st.text("______________________________________________________________________________________________")
    kernel_size = st.slider("Kernal Size", min_value=1, max_value=20, value=0)

    if filter == "Custom 2D Filter":
        processed_image = Custom_Blur(processed_image,kernel_size)
    elif filter == "Built-in Blur":
        processed_image = Blur_function(processed_image,kernel_size)
    elif filter == "Median Blur":
        processed_image = Median_Blur(processed_image, kernel_size)
    elif filter == "Gaussian_Blur":
        processed_image = Gaussian_Blur(processed_image, kernel_size)
    else:
        st.text("Filter not avalible")

    st.image(original_image, caption="★ Original Image ★",width=None)
    st.image(processed_image, caption="★ Blured Image ★")

if __name__ == "__main__":
    main_opration()