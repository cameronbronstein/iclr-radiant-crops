FROM python:3.7-slim

LABEL maintainer="Cameron Bronstein <cambostein@gmail.com>"

WORKDIR /app

COPY ./docker-requirements.txt /app/docker-requirements.txt

RUN pip --no-cache-dir install -r docker-requirements.txt

COPY . /app

VOLUME /app/submissions

CMD ["/bin/bash"]