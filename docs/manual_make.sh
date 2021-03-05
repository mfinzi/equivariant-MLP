#!/bin/bash
jupytext --sync ./notebooks/*
sphinx-build -b html . ./build/html