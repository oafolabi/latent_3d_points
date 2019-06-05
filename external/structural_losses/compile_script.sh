#!/bin/bash

TF_INC="$(python ./print_tf_path.py)"

export TF_INC
make 
