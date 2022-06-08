FROM ubuntu:latest

ENV TZ=America/Los_Angeles
RUN echo "root:test" | chpasswd
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update

RUN apt-get install -y openssh-server
RUN apt-get install -y openssh-client
RUN apt-get install -y python3.8-venv
RUN apt-get install -y curl

RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

ENTRYPOINT service ssh start && bash

EXPOSE 80
EXPOSE 22