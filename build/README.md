# Legacy starter Docker setup

This Dockerfile is retained from the starter and is not the environment used
for the submitted results. It targets TensorFlow 2.5 / Python 3.8, while the
verified Udacity runtime uses TensorFlow 2.3 / Python 3.7. It also installs the
unpinned TensorFlow Models default branch, so it is not a reproducible lockfile.
Use the Udacity reproduction procedure in the project README for this project.
A clean Docker build and compatibility with these scripts have not been
verified; the historical commands below are provided for reference.

## Requirements

* NVIDIA GPU with the latest driver installed
* docker / nvidia-docker

The starter originally reported this build as tested with Nvidia Drivers 460.91.03 and CUDA 11.2 on a Ubutun 20.04 machine.
Please update the base image if you plan on using older versions of CUDA.

## Build
Build the image with:
```
docker build -t project-dev -f Dockerfile .
```

Create a container with:
```
docker run --gpus all -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ --network=host -ti project-dev bash
```
and any other flag you find useful to your system (eg, `--shm-size`).

## Set up

Once in container, you will need to auth using:
```
gcloud auth login
```

## Debug
* Follow this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation) if you run into any issue with the installation of the
tf object detection api

## Updating the instructions
Feel free to submit PRs or issues should you see a scope for improvement.
