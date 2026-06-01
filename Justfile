default:
    just --list

build:
    dune build
    cc -O3 -ffast-math -march=native -o c/snake c/snake.c

play:
    dune exec --profile release bin/main.exe -- play

train n="" envs="":
    dune exec --profile release bin/main.exe -- train {{n}} {{envs}}

train-c n="50" envs="64":
    cc -O3 -ffast-math -march=native -o c/snake c/snake.c
    ./c/snake train {{n}} {{envs}}

bench-train-c n="50" envs="64":
    cc -O3 -ffast-math -march=native -o c/snake c/snake.c
    ./c/snake bench-train {{n}} {{envs}}

bench:
    dune exec --profile release bin/main.exe -- bench

rollout-bench domains="1" envs="256" steps="64" iters="20":
    dune exec --profile release bin/main.exe -- rollout-bench {{domains}} {{envs}} {{steps}} {{iters}}

rollout-bench-c envs="256" steps="64" iters="20":
    cc -O3 -ffast-math -march=native -o c/snake c/snake.c
    ./c/snake rollout-bench {{envs}} {{steps}} {{iters}}

test:
    dune test
