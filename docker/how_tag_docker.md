docker container commit <sha> s1sharp/style_torch_train:latest
docker image tag torch_env:latest s1sharp/style_torch_train:latest
docker image push s1sharp/style_torch_train:latest