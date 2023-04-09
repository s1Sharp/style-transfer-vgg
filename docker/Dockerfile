FROM nvidia/cuda:11.8.0-base-ubuntu22.04

LABEL maintainer.email="maksim.carkov.201300@gmail.com"
LABEL maintainer.git="s1Sharp"

# open for ssh connection
EXPOSE 22

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    git \
    bzip2 \
    libx11-6 \
    wget \
    python3 \
    python3-pip

# Create a working directory.
RUN mkdir /app
WORKDIR /app

COPY ./src/requirements_dev.txt .

# update packages
RUN pip3 install -r requirements_dev.txt