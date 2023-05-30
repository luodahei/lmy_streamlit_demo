import cv2
import numpy as np
from pathlib import Path
import streamlit as st
from inference_onnx import onnxModel,vis
import os
from PIL import Image


'''init model'''
model_path = "model_files/pine.onnx"
model = onnxModel(model_path,
                  class_names=['pine_good','pine_bad'],
                  keep_names=['pine_good','pine_bad'],
                  )
os.makedirs('data/images',exist_ok=True)
os.makedirs('data/result',exist_ok=True)

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


if __name__ == '__main__':

    st.title('LMY Streamlit App')

    source = ("图片检测", "视频检测")
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                image_dir = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_dir = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    if is_valid:
        print('valid')
        if st.button('开始检测'):

            '''model pred and show result'''
            img = cv2.imdecode(np.fromfile(image_dir,dtype=np.uint8),-1)
            boxes = model.run(img, score_thr=0.5)
            show_img = vis(img,boxes)
            cv2.imwrite('data/result/show.jpg',show_img)

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    st.image('data/result/show.jpg')
                    # for img in os.listdir(get_detection_folder()):
                        # st.image(str(Path(f'{get_detection_folder()}') / img))
                    st.balloons()
            else:
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / vid))

                    st.balloons()
