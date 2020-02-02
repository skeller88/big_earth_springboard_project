# Image Dockerfile:
# https://github.com/tensorflow/tensorflow/blob/d73faf5fbb7c8bfbf96cc6334111e4352f209e82/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile
FROM continuumio/miniconda:4.7.12

WORKDIR /app

COPY ./data_science/jupyter_tensorflow_notebook/environment.yml ./environment.yml
RUN conda env create -f ./environment.yml

RUN echo "source activate app-env" >> ~/.bashrc
ENV PATH /opt/conda/envs/app-env/bin:$PATH

COPY ./data_science ./data_science
COPY ./data_engineering ./data_engineering
COPY ./.gcs ./.gcs

# RUN mkdir -p /host_mnt
# RUN chown newuser /host_mnt
# USER newuser

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["bash", "-c", "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root \
--NotebookApp.password='sha1:53b6a295837d:d096b7b1797ebe5bb5f5ecc355659d760281e343'"]
