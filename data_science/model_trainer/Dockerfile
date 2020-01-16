FROM python:3.6
FROM base as builder

ARG filedir

RUN echo "filedir" $filedir

RUN mkdir /install
WORKDIR /install

COPY data_engineering/archive_transferrer_from_big_earth_to_gcs/requirements.txt /requirements.txt
RUN pip install --install-option="--prefix=/install" -r /requirements.txt

FROM base
COPY --from=builder /install /usr/local
COPY ./data_science /app/data_science

WORKDIR /app

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "./data_science/model_trainer/main.py"]