docker build -f ./docker/Dockerfile -t style_torch_train .
docker run --env-file config/.env --gpus all -it -v ${PWD}:/home style_torch_train