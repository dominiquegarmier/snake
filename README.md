# snake

Minimal zero-dependency C PPO snake experiment on an 8x8 board.

```sh
just train       # live throttled ASCII viz, runs until interrupted
just train 60    # train for 60 seconds
just bench 20    # no viz, useful for throughput checks
```

Layout is intentionally flat: `snake.c` plus `Justfile`.

The agent sees only board cells and current direction. It samples three relative actions
left/straight/right and trains a small actor-critic MLP with PPO. There is no path finder,
teacher, action mask, safety filter, Hamiltonian prior, food-placement curriculum, or external ML
library.
Reward is food, death, a small living cost, and idle timeout only; visited cells are tracked only
for reporting.

Stats report mean score, best score, peak snake length, length coverage, visit coverage, and the
share of eval games reaching at least 20 cells, which is the 30% coverage threshold on 8x8.

On this machine, `just bench 120` verified 50.0% mean peak-length coverage and 95.5% mean visit
coverage after 120 seconds.
