FROM jupyter/tensorflow-notebook

COPY ./data_science /app/data_science
WORKDIR /app

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"