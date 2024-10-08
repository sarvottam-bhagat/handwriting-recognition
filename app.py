import streamlit as st
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Set page configuration
st.set_page_config(page_title="Quiz Handwriting Recognition", layout="wide")

st.title("Test Quiz Platform with Handwriting Recognition")

st.write("""
This platform allows you to upload images of handwritten answers to quiz questions.
You can choose between two methods: Optical Character Recognition (OCR) using Tesseract or advanced recognition using a deep learning model.
""")

# Load the TrOCR model and processor
@st.cache_resource
def load_trocr_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

processor, model = load_trocr_model()

# Sample quiz question
st.header("Sample Question:")
st.write("Explain the process of photosynthesis in plants.")

# Image upload (with non-empty label)
uploaded_file = st.file_uploader("Upload your handwritten answer as an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Option to choose between Tesseract OCR and Advanced Deep Learning Model
    recognition_method = st.radio(
        "Choose a method for handwriting recognition:",
        ("Tesseract OCR", "Advanced Deep Learning (TrOCR)")
    )

    if st.button("Recognize Handwriting"):
        st.write("Performing OCR... Please wait.")
        
        # Tesseract OCR method
        if recognition_method == "Tesseract OCR":
            try:
                text = pytesseract.image_to_string(image)
                st.subheader("Recognized Text (Tesseract OCR):")
                st.text_area("Tesseract OCR Result", value=text, height=200, label_visibility="collapsed")
            except Exception as e:
                st.error(f"An error occurred during Tesseract OCR: {str(e)}")
        
        # Advanced Deep Learning (TrOCR) method
        elif recognition_method == "Advanced Deep Learning (TrOCR)":
            try:
                # Preprocess image and generate predictions
                pixel_values = processor(images=image, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                st.subheader("Recognized Text (Advanced Deep Learning):")
                st.text_area("TrOCR Result", value=generated_text, height=200, label_visibility="collapsed")
            except Exception as e:
                st.error(f"An error occurred during TrOCR recognition: {str(e)}")

st.write("""
Note: The accuracy of handwriting recognition may vary depending on the clarity of the handwriting and the quality of the uploaded image.
For best results, ensure that the handwriting is clear and the image is well-lit and in focus.
""")
