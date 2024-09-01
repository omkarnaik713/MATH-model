FROM python
RUN pip install --upgrade pip 
RUN apt-get update && apt-get install -y libhdf5-dev curl
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly -y
RUN source $HOME/.cargo/env
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python math_1.py 
