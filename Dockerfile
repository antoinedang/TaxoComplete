FROM continuumio/anaconda3

# Install all needed deps
RUN apt-get update
RUN apt-get install -y --no-install-recommends git
RUN apt-get install -y vim

COPY requirements requirements

RUN conda create -y -n taxocomplete python=3.9
SHELL ["conda", "run", "--no-capture-output", "-n", "taxocomplete", "/bin/bash", "-c"]

RUN conda install -y -c conda-forge cudatoolkit=11.1 cudnn=8.1.0
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN conda install -y -c dglteam dgl-cuda11.1
RUN pip install -r requirements
RUN pip install transformers==4.29.2 safetensors==0.3.0
RUN pip install chardet
RUN pip install matplotlib

RUN echo "conda activate taxocomplete" >> ~/.bashrc

RUN apt-get autoremove -y
RUN apt-get clean
RUN rm requirements && rm -rf /root/.cache/pip && rm -rf /var/lib/apt/lists/*