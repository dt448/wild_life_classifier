import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image

st.title('Image Classifier')
st.text('Upload the Image')
model = pickle.loaD(open("clf.p", rb))

uploaded_file = st.file_uploader("Choose an image ...", type ="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption = 'Uploaded Image')
    
