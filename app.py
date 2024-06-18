from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0: 'Thaumatophyllum bipinnatifidum',  
       1: 'Thaumatophyllum xanadu',
       2: 'Hederaceum oxycardium ', 
       3: 'Hederaceum oxycardium Brazil'}

model = load_model('MobileNetV2-philodendronbaru-94.99.h5')
model.compile()  # Compile the model to ensure metrics are built

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    p = np.argmax(model.predict(i), axis=-1)
    return dic[int(p[0])]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("philoden.html")
    
@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classification.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename 
        img.save(img_path)
        p = predict_label(img_path)
        return render_template("classification.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
