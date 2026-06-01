# snake

Minimal OCaml PPO snake playground.

Zero runtime dependencies. No ML library. No array math library.

## targets

```sh
just train
just train-c
just bench-train-c
just train 500
just train 500 128
just bench
just rollout-bench 4
just rollout-bench-c
just test
```

## layout

- `lib/game.ml` snake
- `lib/batch.ml` parallel batched simulation with OCaml domains
- `lib/progress.ml` pure ASCII progress bars
- `lib/pp8.ml` 8x8 PPO with a handwritten CNN actor-critic
- `bin/main.ml` `play`, `train`, `bench`

## model

The policy is a handwritten CNN actor-critic: two 3x3 conv layers with 6 filters each, then a
96-unit dense layer feeding policy and value heads. It sees only board cells and direction, samples
all three relative actions, and learns from rollout rewards. A rollout fails after
`min(64, 24 + snake_length)` steps without food. No Hamiltonian prior, teacher, path-finding
features, action mask, safety filter, or food-placement curriculum.

## c path

`c/snake.c` is a separate zero-dependency C implementation of the same 8x8 CNN PPO experiment.
It is currently for speed comparison and does not yet include the async ASCII preview.
