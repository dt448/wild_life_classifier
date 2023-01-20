import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
import sklearn

st.title('British Wild Life Classifier')
st.text('Upload an image to classify')
model = pickle.load(open("clf.p", 'rb'))

uploaded_file = st.file_uploader("Choose an image ...", type ="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption = 'Uploaded Image')
    img = np.array(img)
    
    if st.button('PREDICT'):
        CATEGORIES = ['pigeon', 'daffodil', 'fox']
        st.write('Results...')
        flat_data = []
        img_resized = resize(img,(150,150,3))
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)
        y_out = model.predict(flat_data)
        y_out = CATEGORIES[y_out[0]]
        st.title(f' This is a {y_out}')
        q = model.predict_proba(flat_data)
        st.write("But I am only:")
        for index, item in enumerate(CATEGORIES):
            st.write(f'{q[0][index]*100}% sure it is a {item}')

# Use docker and aws to deploy