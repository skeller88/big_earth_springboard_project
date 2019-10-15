FROM jupyter/tensorflow-notebook

# Install Stackdriver monitoring agent and logging agent:
# https://app.google.stackdriver.com/settings/accounts/agent?project=big-earth-252219
RUN curl -sSO https://dl.google.com/cloudagents/install-monitoring-agent.sh
RUN sudo bash install-monitoring-agent.sh

RUN curl -sSO https://dl.google.com/cloudagents/install-logging-agent.sh
RUN sudo bash install-logging-agent.sh

RUN pip install -r data_science/jupyter_notebook/requirements.txt

COPY ./ /app

WORKDIR /app

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"
