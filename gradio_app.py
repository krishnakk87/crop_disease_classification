
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Load the classes
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
               'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 
               'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
               'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
               'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
               'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# 2. Load TFLite Model
tflite_model_path = 'plant_disease_mobilenetv3.tflite'
if not os.path.exists(tflite_model_path):
    tflite_model_path = '/content/plant_disease_mobilenetv3.tflite'

# Allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_crop_disease(image):
    if image is None:
        return "No image uploaded.", {}
    
    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Apply MobileNetV3 Preprocessing
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Format output labels with confidence scores
    results = {}
    for idx, conf in enumerate(output_data):
        clean_name = class_names[idx].replace('___', ' - ').replace('__', ' ')
        results[clean_name] = float(conf)
    
    # Get top predicted class
    top_idx = np.argmax(output_data)
    top_class = class_names[top_idx].replace('___', ' - ').replace('__', ' ')
    top_confidence = output_data[top_idx] * 100
    
    status = "Pathological disease markers detected on crop foliage!" 
    if "healthy" in top_class.lower():
        status = "The crop foliage appears healthy!"
        
    summary_text = f"Diagnostic Match: {top_class}
Inference Confidence: {top_confidence:.2f}%

Health Assessment: {status}"
    return summary_text, results

# Build Gradio UI
interface = gr.Interface(
    fn=predict_crop_disease,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=[
        gr.Textbox(label="Diagnostic Assessment", show_copy_button=True),
        gr.Label(num_top_classes=3, label="Top Predictions")
    ],
    title="🌱 Crop Disease Diagnostic Assistant",
    description="Upload a leaf image of Tomato, Potato, or Pepper to identify diseases instantly offline using our optimized MobileNetV3 TFLite model.",
    theme="soft"
)

if __name__ == '__main__':
    interface.launch(share=True)
