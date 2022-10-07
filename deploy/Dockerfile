# Install python
FROM python:3.9

# Install dragg
RUN pip install dragg==1.2.3rc0 #--pre
RUN pip install cvxopt

# Copy parent directory
ADD . /dragg-comp-submission

# Install submission
RUN pip install /dragg-comp-submission/.

# Run simulation
WORKDIR  /dragg-comp-submission/sandbox/simulation
CMD python run_player.py -r "redis://redis" & python run_aggregator.py -r "redis://redis"