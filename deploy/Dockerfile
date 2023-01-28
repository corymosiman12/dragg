# Install python
FROM python:3.9

# Install dragg
RUN pip install dragg
RUN pip install cvxopt

# Copy parent directory
ADD . /dragg

# Run simulation
WORKDIR  /dragg-comp-submission/sandbox/simulation
CMD python main.py
