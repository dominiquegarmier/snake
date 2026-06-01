type t = Game.t array

let create ~n ~width ~height ~seed =
  Array.init n (fun i -> Game.create ~width ~height ~seed:(seed + i))

let split_ranges n parts =
  Array.init parts (fun i ->
      let a = i * n / parts in
      let b = (i + 1) * n / parts in
      (a, b))

let step ?(domains = Domain.recommended_domain_count ()) envs policy =
  let n = Array.length envs in
  let domains = max 1 (min domains n) in
  let ranges = split_ranges n domains in
  let run (a, b) =
    let reward = ref 0.0 in
    let done_ = ref 0 in
    for i = a to b - 1 do
      let env = envs.(i) in
      let r = Game.step env (policy env) in
      reward := !reward +. r.reward;
      if r.done_ then (
        incr done_;
        Game.reset env)
    done;
    (!reward, !done_)
  in
  let handles =
    Array.init (domains - 1) (fun i -> Domain.spawn (fun () -> run ranges.(i + 1)))
  in
  let reward, done_ = run ranges.(0) in
  Array.fold_left
    (fun (reward, done_) handle ->
      let r, d = Domain.join handle in
      (reward +. r, done_ + d))
    (reward, done_) handles

let run ?(domains = Domain.recommended_domain_count ()) ~steps envs policy =
  let n = Array.length envs in
  let domains = max 1 (min domains n) in
  let ranges = split_ranges n domains in
  let run_range (a, b) =
    let reward = ref 0.0 in
    let done_ = ref 0 in
    for _ = 1 to steps do
      for i = a to b - 1 do
        let env = envs.(i) in
        let r = Game.step env (policy env) in
        reward := !reward +. r.reward;
        if r.done_ then (
          incr done_;
          Game.reset env)
      done
    done;
    (!reward, !done_)
  in
  let handles =
    Array.init (domains - 1) (fun i -> Domain.spawn (fun () -> run_range ranges.(i + 1)))
  in
  let reward, done_ = run_range ranges.(0) in
  Array.fold_left
    (fun (reward, done_) handle ->
      let r, d = Domain.join handle in
      (reward +. r, done_ + d))
    (reward, done_) handles
