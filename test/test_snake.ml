let () =
  let env = Snake.Game.create ~width:8 ~height:8 ~seed:1 in
  assert (Array.length (Snake.Game.observation env) = Snake.Game.observation_size);
  for _ = 1 to 32 do
    ignore (Snake.Game.step env Snake.Game.Forward);
    if not (Snake.Game.alive env) then Snake.Game.reset env
  done;
  let envs = Snake.Batch.create ~n:8 ~width:8 ~height:8 ~seed:10 in
  let _, _ = Snake.Batch.step ~domains:2 envs (fun _ -> Snake.Game.Forward) in
  let model = Snake.Pp8.create ~seed:1 in
  ignore (Snake.Pp8.greedy model env);
  ()
