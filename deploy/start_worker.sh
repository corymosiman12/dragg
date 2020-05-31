#!/bin/bash

# Make sure that all of these other servers are ready before starting the worker node

echo "Waiting for redis to start"
/usr/local/wait-for-it.sh --strict redis:6379

echo "Waiting for mongo to start"
/usr/local/wait-for-it.sh --strict mongo:27017

cd /dragg/dragg
python3 main.py
