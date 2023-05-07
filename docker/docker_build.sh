docker build -f ./docker/Dockerfile -t style_torch_train .
docker run --env-file config/.env --gpus all -it -v $(pwd):/home style_torch_train