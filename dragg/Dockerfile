FROM python:3

WORKDIR /dragg

COPY . /dragg
COPY requirements.txt /dragg
COPY ./deploy/wait-for-it.sh /usr/local/wait-for-it.sh
COPY ./deploy/start_worker.sh start_worker.sh
RUN pip3 install --ignore-installed -r ./requirements.txt \
    && pip3 install .

CMD ["/dragg/start_worker.sh"]
