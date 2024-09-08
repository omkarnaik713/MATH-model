FROM python
RUN apt-get update && apt-get install -y --no-install-recommends libhdf5-dev curl 
RUN pip install --upgrade pip 
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
#RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python math_1.py 
