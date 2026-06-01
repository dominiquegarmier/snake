let sleep seconds = ignore (Unix.select [] [] [] seconds)

let clear () = print_string "\027[2J\027[H"

let greedy env =
  let obs = Snake.Game.observation env in
  if obs.(0) = 0.0 then Snake.Game.Forward
  else if obs.(1) = 0.0 then Snake.Game.Turn_right
  else Snake.Game.Turn_left

let play () =
  let env = Snake.Game.create ~width:20 ~height:12 ~seed:1 in
  for _ = 1 to 400 do
    clear ();
    Printf.printf "score=%d steps=%d\n%s%!" (Snake.Game.score env) (Snake.Game.steps env) (Snake.Game.render env);
    let _ = Snake.Game.step env (greedy env) in
    if not (Snake.Game.alive env) then Snake.Game.reset env;
    sleep 0.05
  done

let show_train_snapshot snapshot status stop =
  let sleep seconds = ignore (Unix.select [] [] [] seconds) in
  let rec loop seed =
    if not (Atomic.get stop) then (
      (match Atomic.get snapshot with
      | None ->
          clear ();
          Printf.printf "snake ppo 8x8\n%s\n\nwaiting for first policy...%!" (Atomic.get status);
          sleep 0.1;
          loop seed
      | Some (version, model) ->
          let env = Snake.Game.create ~width:8 ~height:8 ~seed in
          while
            (not (Atomic.get stop))
            && Snake.Game.alive env
            && env.Snake.Game.since_food < min (8 * 8) (24 + env.Snake.Game.len)
          do
            clear ();
            Printf.printf "snake ppo 8x8\npolicy=%d %s\nrollout score=%d steps=%d\n\n%s%!" version
              (Atomic.get status) (Snake.Game.score env) (Snake.Game.steps env) (Snake.Game.render env);
            ignore (Snake.Game.step env (Snake.Pp8.greedy model env));
            sleep 0.03
          done;
          loop (seed + 1)))
  in
  loop 50_000

let train ?iters ?env_count ?(lr = 0.003) () =
  let snapshot = Atomic.make None in
  let status = Atomic.make "starting..." in
  let stop = Atomic.make false in
  Sys.set_signal Sys.sigint (Sys.Signal_handle (fun _ -> Atomic.set stop true));
  let visual = Domain.spawn (fun () -> show_train_snapshot snapshot status stop) in
  let t0 = Unix.gettimeofday () in
  let model =
    Snake.Pp8.train ~snapshot ~status ~stop ?iters ?env_count ~render_progress:false ~learning_rate:lr ()
  in
  let train_dt = Unix.gettimeofday () -. t0 in
  Atomic.set stop true;
  Domain.join visual;
  let mean, best, mean_steps = Snake.Pp8.eval model in
  clear ();
  Printf.printf "snake ppo 8x8 episodes=128 mean_score=%.2f best=%d mean_steps=%.0f train_seconds=%.3f\n%!" mean
    best mean_steps train_dt

let bench ?domains ?(env_count = 4096) ?(steps = 1_000) () =
  let envs = Snake.Batch.create ~n:env_count ~width:12 ~height:12 ~seed:1 in
  let policy _ = Snake.Game.Forward in
  let t0 = Unix.gettimeofday () in
  Snake.Progress.draw ~current:0 ~total:steps "env_steps=0";
  let reward, done_ =
    match domains with
    | None -> Snake.Batch.run ~steps envs policy
    | Some domains -> Snake.Batch.run ~domains ~steps envs policy
  in
  Snake.Progress.draw ~current:steps ~total:steps
    (Printf.sprintf "env_steps=%d" (Array.length envs * steps));
  let dt = Unix.gettimeofday () -. t0 in
  let n = float_of_int (Array.length envs * steps) in
  Printf.printf "\nenv_steps=%.0f seconds=%.3f steps_per_second=%.0f reward=%.1f done=%d\n%!" n dt (n /. dt)
    reward done_

let rollout_bench ?(domains = 1) ?(env_count = 256) ?(steps = 64) ?(iters = 20) () =
  let model = Snake.Pp8.create ~seed:7 in
  let envs = Array.init env_count (fun i -> Snake.Game.create ~width:8 ~height:8 ~seed:(10_000 + i)) in
  let t0 = Unix.gettimeofday () in
  for _ = 1 to iters do
    ignore (Snake.Pp8.collect ~domains ~envs ~steps model)
  done;
  let dt = Unix.gettimeofday () -. t0 in
  let n = float_of_int (iters * env_count * steps) in
  Printf.printf "rollout domains=%d envs=%d steps=%d iters=%d samples=%.0f seconds=%.3f samples_per_second=%.0f\n%!"
    domains env_count steps iters n dt (n /. dt)

let () =
  match Array.to_list Sys.argv with
  | [ _; "play" ] -> play ()
  | [ _; "train" ] -> train ()
  | [ _; "train"; iters ] -> train ~iters:(int_of_string iters) ()
  | [ _; "train"; iters; envs ] -> train ~iters:(int_of_string iters) ~env_count:(int_of_string envs) ()
  | [ _; "train"; iters; envs; lr ] ->
      train ~iters:(int_of_string iters) ~env_count:(int_of_string envs) ~lr:(float_of_string lr) ()
  | [ _; "bench" ] -> bench ()
  | [ _; "bench"; domains ] -> bench ~domains:(int_of_string domains) ()
  | [ _; "bench"; domains; envs; steps ] ->
      bench ~domains:(int_of_string domains) ~env_count:(int_of_string envs) ~steps:(int_of_string steps) ()
  | [ _; "rollout-bench" ] -> rollout_bench ()
  | [ _; "rollout-bench"; domains ] -> rollout_bench ~domains:(int_of_string domains) ()
  | [ _; "rollout-bench"; domains; envs; steps; iters ] ->
      rollout_bench ~domains:(int_of_string domains) ~env_count:(int_of_string envs) ~steps:(int_of_string steps)
        ~iters:(int_of_string iters) ()
  | _ ->
      prerr_endline "usage: snake {play|train|bench|rollout-bench}";
      exit 2
