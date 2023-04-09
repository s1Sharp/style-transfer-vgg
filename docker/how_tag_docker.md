docker container commit <sha> s1sharp/torch_env:latest
docker image tag torch_env:latest s1sharp/torch_env:latest
docker image push s1sharp/torch_env:latest