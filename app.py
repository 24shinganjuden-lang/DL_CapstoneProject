import gradio as gr
import numpy as np
from tensorflow import keras
from PIL import Image

CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
IMG_SIZE = (224, 224)

try:
    mnet = keras.models.load_model('models/mobilenetv3_neu.keras')
except:
    mnet = keras.models.load_model('mobilenetv3_neu.keras')

def predict(img):
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype='float32'), 0)
    probs = mnet.predict(arr, verbose=0)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(6)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Upload steel surface image'),
    outputs=gr.Label(num_top_classes=3, label='Predicted defect'),
    title='Steel Surface Defect Classifier',
    description='MobileNetV3Small (transfer learning) trained on the NEU surface defect database.')

if __name__ == '__main__':
    demo.launch(share=True)
