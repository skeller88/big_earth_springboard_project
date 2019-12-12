FROM python:3.6-alpine

ARG SPARK_VERSION=2.4.3
ARG HADOOP_VERSION=2.7

RUN apk --update add wget tar bash

# https://stackoverflow.com/questions/51121875/how-to-run-docker-with-python-and-java
RUN apk --update add --no-cache openjdk8-jre && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
 tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
 rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
 ln -s /spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /spark

ARG SPARK_HOME=/spark
ENV JAVA_HOME=/usr/lib/jvm/java-1.8-openjdk
ENV PYTHONPATH /usr/bin/python:$PYTHONPATH
ENV PATH $SPARK_HOME/python:$PATH
WORKDIR /app
