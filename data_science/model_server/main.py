import json
import os
from pathlib import Path

import flask
import numpy as np
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from tensorflow.keras.models import load_model

app = flask.Flask(__name__)
model = None
stats = None


def image_processor(img, stats):
    img = np.array(img).reshape((120, 120, 3)).astype(np.uint16)
    normalized_img = (img - stats['mean'].values) / stats['std'].values
    return normalized_img


@app.route("/classify", methods=["POST"])
def classify():
    data = json.loads(flask.request.get_json(force=True))
    global stats
    image = image_processor(data['image'], stats)

    global model
    pred_probs = model.predict(np.array([image]))
    return flask.jsonify({
        'is_cloud_probability': pred_probs.tolist()[0][0],
        'success': True
    })


if __name__ == "__main__":
    print("* Loading Keras model...")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/.gcs/big-earth-252219-fb2e5c109f78.json'
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(os.environ.get("GCS_BUCKET"))
    tmp_model_path = "/tmp/tmp_model.h5"
    gcs_model_blob = bucket.blob(os.environ.get("GCS_MODEL_BLOB"))
    gcs_model_blob.download_to_filename(tmp_model_path)
    model = load_model(tmp_model_path)

    tmp_stats_path = "/tmp/tmp_stats.csv"
    gcs_stats_blob = bucket.blob(os.environ.get("GCS_STATS_BLOB"))
    gcs_stats_blob.download_to_filename(tmp_stats_path)
    stats = pd.read_csv(tmp_stats_path)
    stats = stats[stats['data'] == 'all']

    print('Loaded Keras model.')
    app.run(host='0.0.0.0', port=8889)
