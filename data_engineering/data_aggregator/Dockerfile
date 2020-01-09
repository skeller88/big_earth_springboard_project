FROM jupyter/tensorflow-notebook

ARG filedir

#RUN apk --no-cache add bash curl
#RUN curl -sSO https://dl.google.com/cloudagents/install-monitoring-agent.sh
#RUN bash install-monitoring-agent.sh

RUN mkdir /install
WORKDIR /install

COPY data_engineering/data_aggregator/requirements.txt /requirements.txt
RUN pip install --install-option="--prefix=/install" -r /requirements.txt

COPY --from=builder /install /usr/local
COPY ./data_engineering /app/data_engineering
COPY ./.gcs /app/.gcs

WORKDIR /app

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "/app/data_engineering/data_aggregator/main.py"]