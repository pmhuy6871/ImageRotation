import streamlit as st
import imutils
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def processing_img(file):
    img = load_img(file, target_size=(224, 224))
    arr_img = img_to_array(img)
    arr_img = preprocess_input(arr_img)
    arr_img = tf.expand_dims(arr_img, axis=0)
    return arr_img

def predict(model, image):
    prediction = model.predict(image)
    pred = np.argmax(prediction)
    return pred

def rotate_image(image, angle):
    rotated_angle = 360 - angle
    rotated_image = imutils.rotate_bound(image, rotated_angle)

    # Normalize the rotated image data to the range [0.0, 1.0]
    rotated_image = rotated_image.astype("float") / 255.0

    return rotated_image, rotated_angle

def main():
    st.title("Welcome to My Project :sunglasses:")
    st.header("Deep Learning: Image Classification and Rotation", divider='rainbow')

    st.markdown('''### **Mô tả**:
    Dataset gồm 5 vật: **Bottle, Headphone, Mouse, Pen, Sneaker**.
    Mô hình phân lớp: truyền ảnh để xác định 1 trong 5 đối tượng trên.
    Mô hình xác định góc quay: truyền ảnh để xác định góc quay (0, 90, 180, 270)
    ''')

    file = st.file_uploader('**Chọn ảnh tải lên**', type=['jpg'])
    if file:
        image = load_img(file, target_size=(224, 224))
        st.write("**Bạn đã chọn ảnh:**", file.name)
        st.image(image)

        option_ls = ['Mô hình phân lớp', 'Mô hình tìm góc quay']
        option = st.selectbox(label='**Lựa chọn mô hình**', options=option_ls)

        new_image = processing_img(file)

        classes_path = 'E:/DL/5Thing_Res.hdf5'
        rotation_path = 'E:/DL/5Thing_Rotation.hdf5'
        model_classes = load_model(classes_path)
        model_rotation = load_model(rotation_path)
        class_labels = ['Bottle', 'Headphone', 'Mouse', 'Pen', 'Sneaker']
        rotation_labels = [0, 90, 180, 270]

        if st.button(label='Dự đoán', type='primary'):
            if option == 'Mô hình phân lớp':
                pred_cl = predict(model_classes, new_image)
                st.write('**Dự đoán**:', class_labels[pred_cl])
            elif option == 'Mô hình tìm góc quay':
                pred = predict(model_rotation, new_image)
                angle = rotation_labels[pred]
                st.write('**Dự đoán góc quay**:', angle)

                # Rotate the image to 0 degrees and normalize
                rotated_image, rotated_angle = rotate_image(img_to_array(image), angle)
                st.image(rotated_image, caption=f"Rotated by {rotated_angle} degrees")

if __name__ == "__main__":
    main()