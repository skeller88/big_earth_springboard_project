# USAGE
# Start the server:
# 	python app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'

# import the necessary packages
import csv
import json
import os
from pathlib import Path

import flask
import numpy as np
import pandas as pd
from google.cloud import storage
from tensorflow.keras.models import load_model

# initialize our Flask application and the Keras model

app = flask.Flask(__name__)
model = None
stats = None


def image_processor(img, stats):
    if len(img) != 120 * 120 * 3:
        print(f"Expected image shape of {120 * 120 * 3}, got {len(img)}")
        return None
    img = np.array(img).reshape(120, 120, 3).astype(np.uint16)
    normalized_img = (img - stats['mean'].values) / stats['std'].values
    return normalized_img


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get("image"):
            # read the image in PIL format
            image = json.loads(flask.request.get("image"))

            global stats
            # preprocess the image and prepare it for classification
            image = image_processor(image, stats)

            global model
            pred_prob = model.predict(image)
            data['is_cloud_probability'] = pred_prob
            # indicate that the request was a success
            data["success"] = True

        # return the data dictionary as a JSON response
        return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...")
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(os.environ.get("GCS_BUCKET"))
    tmp_model_path = Path.home() / "tmp" / "tmp_model.h5"
    gcs_model_blob = bucket.blob(os.environ.get("GCS_MODEL_BLOB"))
    gcs_model_blob.download_to_filename(str(tmp_model_path))
    model = load_model(tmp_model_path)

    tmp_stats_path = Path.home() / "tmp" / "tmp_stats.csv"
    gcs_stats_blob = bucket.blob(os.environ.get("GCS_STATS_BLOB"))
    gcs_stats_blob.download_to_filename(str(tmp_stats_path))


    with open(tmp_stats_path) as tmp_stats_file_obj:
        reader = csv.DictReader(tmp_stats_file_obj, fieldnames=[])
        stats = reader.read
    app.run(host='0.0.0.0')
