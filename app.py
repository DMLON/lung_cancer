import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from yolov5_model import Yolo5Detector

##########
##### Set up sidebar.
##########

# Add in location to select image.

@st.cache(allow_output_mutation=True)
def load_model():
    model = Yolo5Detector("best.pt",device="cpu")
    return model

model = load_model()

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)


## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)
model.conf_thres = confidence_threshold
model.iou_thres = overlap_threshold
##########
##### Set up main app.
##########

## Title.
st.write('# Cancer Detection - Object Detection')

## Pull in default image or user-selected image.
if uploaded_file is None:
    # Default image.
    url = 'https://www.cancer.gov/sites/g/files/xnrzdm211/files/styles/cgov_social_media/public/cgov_image/media_image/900/100/files/adenocarcinoma-ct-scan-article.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
    image = Image.open(uploaded_file)
    image = image.convert('RGB')

## Subtitle.
st.write('### Inferenced Image')

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='PNG')

image_numpy = np.array(image)
# image = np.expand_dims(image, axis=0)
image_annotated, predictions = model.detect_return_img(image_numpy)

# Display image.
st.image(image_annotated,
         use_column_width=True)

## Generate list of confidences.
confidences = [box['score'] for box in predictions]

## Summary statistics section in main app.
st.write('### Summary Statistics')
st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
confidences_numpy = []
for conf in confidences:
    confidences_numpy.append(conf.cpu().detach().numpy())
st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences_numpy),4))}')

## Histogram in main app.
st.write('### Histogram of Confidence Levels')
fig, ax = plt.subplots()
ax.hist(confidences_numpy, bins=10, range=(0.0,1.0))
st.pyplot(fig)

## Display the JSON in main app.
st.write('### JSON Output')
st.write(predictions)