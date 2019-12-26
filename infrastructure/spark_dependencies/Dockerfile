FROM continuumio/miniconda3:latest

ARG
RUN conda install -c conda-forge conda-pack

conda pack -n my_env

gsutil cp gs://dataproc-initialization-actions/presto/presto.sh gs://my-bucket/