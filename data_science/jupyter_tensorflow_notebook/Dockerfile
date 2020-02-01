FROM jupyter/tensorflow-notebook

# Maybe install needed tf dependencies?
# https://github.com/tensorflow/tensorflow/issues/34329

COPY data_science/jupyter_tensorflow_notebook/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-cache-dir

COPY ./data_science /app/data_science
COPY ./data_engineering /app/data_engineering
COPY ./.gcs /app/.gcs

# RUN mkdir -p /host_mnt
# RUN chown newuser /host_mnt
# USER newuser

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"