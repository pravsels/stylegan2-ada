
ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:20.10-tf1-py3
FROM $BASE_IMAGE

ENV SGAN_WS=/workspace

RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install h5py==2.9.0
RUN pip install imageio==2.9.0
RUN pip install imageio-ffmpeg==0.4.2
RUN pip install tqdm==4.49.0

COPY ./Rel_5.0.0 /Rel_5.0.0

RUN mkdir /nbis

RUN cd /Rel_5.0.0 && \
	./setup.sh /nbis --without-X11 --64 && \
	make config && \
	make it && \
	make install

WORKDIR $SGAN_WS

CMD ["bash"]
