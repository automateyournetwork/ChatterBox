FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo \
 && sudo apt-get update -y

RUN echo "==> Installing Python3 and pip ...." \  
 && apt-get install python3 -y \
 && apt install python3-pip -y \
 && apt install openssh-client -y

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install langchain requirements.." \
  && pip install -U langchain langchain-openai langchain-community \
  && pip install chromadb \
  && pip install openai \
  && pip install tiktoken

RUN echo "==> Install jq.." \
  && pip install jq

RUN echo "==> Install streamlit.." \
  && pip install streamlit

RUN echo "==> Install aio.." \
  && pip install aiofiles \
  && pip install aiohttp

COPY /chatterbox /chatterbox/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]