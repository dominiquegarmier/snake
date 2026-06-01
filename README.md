# snake

```sh
just train     # train
just train 60  # train 60s, the preview
just bench 60
```

```
snake ppo c 8x8  [TRAIN]   Ctrl+C quit
iter 53     time    5.6s   envs 256       78182 sample/s
score  0.86   best 3    len  3.9/64   6.0%   p30   0%
visit  37.9%   p30  74%   steps 40

+--------+
|........|
|........|
|......*.|
|........|
|.o......|
|.o......|
|.#......|
|........|
+--------+

# head  o body  * food
```
