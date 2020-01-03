FROM jupyter/tensorflow-notebook

# Install Stackdriver monitoring agent and logging agent:
# https://app.google.stackdriver.com/settings/accounts/agent?project=big-earth-252219
# RUN sudo apt-get install stackdriver-agent

COPY ./data_science/jupyter_notebook/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY ./data_science /app

WORKDIR /app

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"
