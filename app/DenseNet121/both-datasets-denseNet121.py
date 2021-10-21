# %%
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os

# %%
import cv2
import numpy as np
# from tensorflow.keras import layers
# from tensorflow.keras.applications import DenseNet121
# # from tensorflow.keras.callbacks import Callback, ModelCheckpoint
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import load_model
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# from sklearn import metrics
# import tensorflow as tf

# %%
# def build_model(pretrained):
#     model = Sequential([
#         pretrained,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(1, activation='sigmoid')
#     ])

#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam(),
#         metrics=['accuracy']
#     )
    
#     return model

# # %%
# densenet = DenseNet121(
#     weights=None,
#     include_top=False,
#     input_shape=(224,224,1)
# )
# model = build_model(densenet)

# %%
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model('real-and-fake-face-and-140k-images_densenet.h5')

# %%

img = load_img(r'real_01081.jpg', target_size=(224, 224))
img = img_to_array(img)
img = img / 255
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.expand_dims(gray,axis=0)

# %%
y_pred = model.predict(gray)

# %%
y_pred[0][0]

# %%
if y_pred > 0.5:
    ans = 'REAL'
else: 
    ans = 'FAKE'

# %%
statement = f'The probability of this image being real is {y_pred[0][0]*100:.3f}%.\nThe probability of this image being fake is {(1-y_pred[0][0])*100:.3f}%.\nTherefore, this image is ' + ans + '.'

# %%
print(statement)

# %%


# %%
