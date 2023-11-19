from flask import Flask, request
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']
    img_data = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_data))
    
    pred, pred_idx, probs = learn_inf.predict(img)
    result = {learn_inf.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}
    
    return result

if __name__ == '__main__':
    app.run(debug=True)
