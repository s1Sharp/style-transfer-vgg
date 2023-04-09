docker build -f ./docker/Dockerfile -t torch_env .
docker run --gpus all -it -v ${PWD}:/home torch_env