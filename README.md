# STONK
![STONKS](/images/STONKS.png)

Stocks
Trading
Optimized
Network
Knowledge

Recommender system for stocks trading. Recommends every day to buy, sell or do nothing in order to maximize the proffit.


## Quickstart

To start the docker container execute the following command

```sh
$ ./bin/start [-n <string>] [-t <tag-name>] [--sudo] [--build]
```

```
<tag-name> = cpu | devel-cpu | gpu
```

For example:

```sh
$ ./bin/start -n myContainer -t gpu --build
```

Once the docker container is running it will execute the contents of the /bin/execute file.

You can execute

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```
to access the running container's shell.

## Datasets

I used New York Stock Exchange dataset made by Dominik Gawlik
https://www.kaggle.com/dgawlik/nyse

## Models & Techniques

The model architecture is comprised of a Multi-head attention layer and a LSTM layer. The model was trained with reinforcement learning, in particular PPO was used.
