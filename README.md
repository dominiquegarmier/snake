# snake

Zero-dependency C PPO snake on 8x8.

```sh
just train     # run until interrupted
just train 60  # run 60 seconds
just bench 20  # no viz
```

No pathfinder, teacher, action mask, safety filter, curriculum, or priors. The policy sees only
board cells and direction. Reward is food, death, idle timeout, and a small living cost.
