from fastapi import FastAPI, File
from PIL import Image
from io import BytesIO
import json
import numpy as np
import logging
import time
import requests
import uvicorn
from keras_preprocessing import image


app = FastAPI()

@app.get('/')
async def default():
    return

@app.get('/ping/')
async def ping():
    """
    ping method for api call
    Returns:
        200
    """
    return


@app.post("/analyse/")
async def food_analysis(file: bytes = File(...)):
    """
    Handles image file path and urls

    Args:
        file: file in binary format

    Returns:
        json string of classes and embedding lists, i.e: {"cls": [str(cls)], "embedding": [embedding.tolist()]}
    """
    try:
        start_time = time.time()
        img = Image.open(BytesIO(file)).resize((224,224)).convert("RGB")
        # np_img = np.array(img.convert('RGB'), dtype=np.uint8)

        np_img = image.img_to_array(img) / 255.

        # this line is added because of a bug in tf_serving < 1.11
        img = np_img.astype('float16')

        # Creating payload for TensorFlow serving request
        payload = {
            "instances": [{'input_image': img.tolist()}]
        }

        # Making POST request
        r = requests.post('http://localhost:9000/v1/models/ImageClassifier:predict', json=payload)

        # Decoding results from TensorFlow Serving server
        pred = json.loads(r.content.decode('utf-8'))
        categories = [
            'healthy', 'junk', 'dessert', 'appetizer', 'mains', 'soups', 'carbs', 'protein', 'fats', 'meat'
        ]
        results = np.array(pred['predictions'])[0]
        preds = {}
        for i in range(len(categories)):
            preds[categories[i]] = int(results[i] * 100 // 1)
        # Returning JSON response to the frontend
        print(f"processed time: {round(time.time() - start_time, 4)}")
        return {'result': preds}

    except Exception as e:
        return {'result': 'error ocused: '+str(e)}


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8050,
        log_level="info",
        workers=1,
        reload=False
    )