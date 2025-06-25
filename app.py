# from flask import Flask, request, jsonify
# from flask_cors import CORS
from model.main import caption_model, greedy_algorithm, vectorization
import gradio as gr

# app = Flask(__name__)
# CORS(app)  # Enables CORS to allow frontend from different origins

# @app.route('/caption', methods=['POST'])
# def caption_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     file = request.files['image']  # Get the uploaded image
#     print("Received file:", file.filename)

#     try:
#         # Pass the file stream to the greedy algorithm
#         caption = greedy_algorithm(file.stream, vectorization, caption_model, True)
#         print("Generated caption:", caption)
#         return jsonify({'caption': caption})
#     except Exception as e:
#         print("Error generating caption:", str(e))
#         return jsonify({'error': str(e)}), 500

def caption_image_gradio(img):
    return greedy_algorithm(img, vectorization, caption_model, False)
# if __name__ == '__main__':
#     app.run(debug=True)
gr.Interface(fn=caption_image_gradio, 
             inputs=gr.Image(type="pil"), 
             outputs="text").launch(share=True)

