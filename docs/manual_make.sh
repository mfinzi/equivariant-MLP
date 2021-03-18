#!/bin/bash
# jupytext --sync ./notebooks/*
./notebooks/merge_nbs.sh
sphinx-build -b html . ./build/html