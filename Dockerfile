# Set base image (host OS)
FROM python:3.8

# Install CMake
RUN apt-get update && \
  apt-get --yes install cmake

# Install mgclient
RUN apt-get install -y git cmake make gcc g++ libssl-dev && \
  git clone https://github.com/memgraph/mgclient.git /mgclient && \
  cd mgclient && \
  git checkout dd5dcaaed5d7c8b275fbfd5d2ecbfc5006fa5826 && \
  mkdir build && \
  cd build && \
  cmake .. && \
  make && \
  make install

# Install pymgclient
RUN git clone https://github.com/memgraph/pymgclient /pymgclient && \
  cd pymgclient && \
  python3 setup.py build && \
  python3 setup.py install

# Set the working directory in the container
WORKDIR /python

# Copy the dependencies file to the working directory
#COPY requirements.txt .

# Install dependencies
#RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Copy module to memgraph
ADD python/graph_coloring.py /usr/lib/memgraph/query_modules

# Command to run on container start
CMD [ "python" ]
