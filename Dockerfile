FROM python:3

RUN pip install numpy matplotlib plotly scipy sklearn h5py PyYAML

ADD consensus.py pre_processing.py ensemble_clustering.py /

CMD [ "python", "./consensus.py" ]
