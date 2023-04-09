docker build -f ./docker/Dockerfile -t torch_env .
docker run --gpus all -it -v $(pwd):/home torch_env