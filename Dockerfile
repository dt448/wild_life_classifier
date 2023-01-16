# syntax=docker/dockerfile:1
FROM python:3.8-slim

# Install app dependencies
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-venv

WORKDIR /wildlife-classifier

RUN python3 -m venv wildlife-classifier-env
RUN wildlife-classifier-env/bin/pip install streamlit scikit-image scikit-learn

COPY app.py /wildlife-classifier/
COPY clf.p /wildlife-classifier/

EXPOSE 8501
CMD . ./wildlife-classifier-env/bin/activate && streamlit run app.py
