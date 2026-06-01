default:
    just --list

build:
    cc -O3 -ffast-math -march=native -flto -Wall -Wextra -o snake snake.c

train seconds="0" envs="256":
    cc -O3 -ffast-math -march=native -flto -Wall -Wextra -o snake snake.c
    ./snake train {{seconds}} {{envs}}

bench seconds="30" envs="256":
    cc -O3 -ffast-math -march=native -flto -Wall -Wextra -o snake snake.c
    ./snake bench {{seconds}} {{envs}}
