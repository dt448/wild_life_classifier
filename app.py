import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
import sklearn
# from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.resnet_v2 import preprocess_input

# st.title('British Wild Life Classifier')
# st.text('Upload an image to classify')
# model = pickle.load(open("clf.p", 'rb'))

# uploaded_file = st.file_uploader("Choose an image ...", type ="jpg")
# if uploaded_file is not None:
#     img = Image.open(uploaded_file)
#     st.image(img, caption = 'Uploaded Image')
#     img = np.array(img)
    
#     if st.button('PREDICT'):
#         CATEGORIES = ['pigeon', 'daffodil', 'fox']
#         st.write('Results...')
#         flat_data = []
#         img_resized = resize(img,(150,150,3))
#         flat_data.append(img_resized.flatten())
#         flat_data = np.array(flat_data)
#         y_out = model.predict(flat_data)
#         y_out = CATEGORIES[y_out[0]]
#         st.title(f' This is a {y_out}')
#         q = model.predict_proba(flat_data)
#         st.write("But I am only:")
#         for index, item in enumerate(CATEGORIES):
#             st.write(f'{q[0][index]*100}% sure it is a {item}')

# ResNet Model:

model = pickle.load(open("resnet_model_v1.p", 'rb'))

def load_image(image):
    image = image.resize((224,224))
    img_array = np.array(image)#/255 # a normalised 2D array                
    img_array = img_array.reshape(-1, 224, 224, 3)   # to shape as (1, 224, 224, 3)
    return img_array

uploaded_file = st.file_uploader("Choose an image ...", type ="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption = 'Uploaded Image')
    img = np.array(img)
    
    if st.button('PREDICT'):
        # CATEGORIES = ['pigeon', 'daffodil', 'fox']
        CATEGORIES = ["fox", "pigeon", "daffodil", "badger","bank vole", "barbastelle bat", "bechstein's bat", "field vole", "lynx", "otter", "pine marten"]
        st.write('Results...')
        flat_data = []
        img = load_image(Image.open(uploaded_file))
        # img_resized = resize(img,(224,224,3))
        # flat_data.append(img_resized.flatten())
        # flat_data = np.array(flat_data)

        # my_image = img_to_array(my_image)
        # my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
        my_image = preprocess_input(img)

        #make the prediction
        prediction = model.predict(my_image)
        print("predictions:",prediction)

        pred = np.argmax(prediction, axis=-1)

        # y_out = model.predict(flat_data)
        y_out = CATEGORIES[pred[0]]
        st.title(f' This is a {y_out}')

        # q = model.predict_proba(flat_data)
        st.write("But I am only:")
        for index, item in enumerate(CATEGORIES):
            st.write(f'{prediction[0][index]*100}% sure it is a {item}') 