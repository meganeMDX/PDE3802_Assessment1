import streamlit as st
import pandas as pd

# Styling section
st.html("""
<style>
    .stApp{
        background-color: #d6e6f2
    }
    .st-key-mainApp{
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .st-key-my-custom-box {
        background-color: #9bbeff;
        justify-content: center;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .st-key-analyse-btn button{
        background-color: #006eef;
        color: white !important;
    }
    .st-key-btn-img button{
                background-color: #4CAF50;
                color: white !important;
    }
    .st-key-img-drop [data-testid="stFileUploaderDropzone"] {
        background-color: #eaf3ff;
        border: 2px dashed #006eef;
        border-radius: 12px;
        padding: 40px;
    }
    .st-key-btn-cam button{
            background-color: #4CAF50;
            color: white !important;
        }
    .st-key-results{
        background-color: #8db4d6;
        align-self:center !important;
        position: center;
        border-radius:10px;
        width: 400px;
    }
</style>
""")

#code for main interface
with st.container(key="mainApp"):
    st.title("Office items organising robot", text_alignment="center")

    #array to create grid for office items
    with st.container(key="my-custom-box"):
        st.header("Detect the following Office Items with our Trained Model")
        items = ["Laptop", "Keyboard", "Scissors", "Notebook", "Binder", "Bottle", "Phone", "Bin"]
        cols = st.columns(4) 

        #grid to display buttons
        for i, item in enumerate(items):
            if cols[i % 4].button(item, key=f"btn-{item}", width="stretch"):
                st.session_state["selected_item"] = item

    if "selected_item" in st.session_state:
        st.write(f"Selected: **{st.session_state['selected_item']}**")


    #creating buttons for upload image or use live camera
    st.header("Provide image for classification", text_alignment="center")
    with st.container(horizontal=True):
        if st.button("Upload an Image", key="btn-img",  width="stretch"):
            st.session_state['mode'] = 'upload'

        if st.button("Use live camera",key="btn-cam",  width="stretch"):
            st.session_state['mode'] ='camera'

    mode= st.session_state.get('mode')
    if mode == 'upload':
        #using st. file_uploader to create to upload images
        fileUploads = st.file_uploader(
            "Upload or drag and drop an image", accept_multiple_files=True, type=["jpg", "png"], key="img-drop"
        )
        for fileUpload in fileUploads:
            st.image(fileUpload)
    elif mode=='camera':
        #using st.camera_input for live front camera input
        camPermission = st.toggle("Slide to turn on camera")
        liveImage = st.camera_input("Use live camera", disabled=not camPermission)
        if liveImage:
            st.image(liveImage)
            
    with st.container():
        #section for image analysis when analyse image is clicked
        if st.button("Analyse image", key="analyse-btn",  width="stretch"):
            with st.container(key="ImgAnalysis"):
                st.header("Image Analysis")
                st.write("Here is the breakdown of the image processing")
        #clearing media button
        st.button("Clear Image", type="primary", width="stretch")

    #classification result section
    with st.container(key='results'):
        st.header("Classification result", text_alignment="center")
        st.subheader("<Category_name>", text_alignment="center")
        st.write("Confidence level: {percentage}%")
        if st.button("View Result", icon_position="center"):
            with st.popover("Open popover", on_change=openPopUp, key="drawer"):
                st.write("This is the popover")
                st.button("Close popover", on_click=closePopUp)
    def openPopUp():
        pass
    def closePopUp():
        pass
#to run the file type streamlit run app.py