let size = 8
let cells = size * size
let channels = 4
let actions = 3
let conv1 = 6
let conv2 = 6
let kernel = 3
let kernel_area = kernel * kernel
let dir_count = 4
let features = (conv2 * cells) + dir_count
let hidden = 96
let active = cells + 1
let lr = ref 0.003
let idle_limit env = min cells (24 + env.Game.len)

external dense_tanh_forward :
  float array -> float array -> float array -> float array -> int -> int -> unit
  = "snake_dense_tanh_forward_byte" "snake_dense_tanh_forward"

external dense_tanh_backward :
  float array ->
  float array ->
  float array ->
  float array ->
  float array ->
  float array ->
  float array ->
  int ->
  int ->
  unit = "snake_dense_tanh_backward_byte" "snake_dense_tanh_backward"

type model = {
  rng : Rng.t;
  wc1 : float array;
  bc1 : float array;
  wc2 : float array;
  bc2 : float array;
  wf : float array;
  bf : float array;
  wp : float array;
  bp : float array;
  wv : float array;
  mutable bv : float;
  mwc1 : float array;
  vwc1 : float array;
  mbc1 : float array;
  vbc1 : float array;
  mwc2 : float array;
  vwc2 : float array;
  mbc2 : float array;
  vbc2 : float array;
  mwf : float array;
  vwf : float array;
  mbf : float array;
  vbf : float array;
  mwp : float array;
  vwp : float array;
  mbp : float array;
  vbp : float array;
  mwv : float array;
  vwv : float array;
  mutable mbv : float;
  mutable vbv : float;
  mutable adam_t : int;
}

type rollout = {
  obs : int array;
  actions : int array;
  old_logp : float array;
  rewards : float array;
  dones : bool array;
  values : float array;
  last_values : float array;
  adv : float array;
  returns : float array;
  n : int;
  envs : int;
}

type work = {
  c1 : float array;
  c2 : float array;
  feat : float array;
  h : float array;
  logits : float array;
}

let make_work () =
  {
    c1 = Array.make (conv1 * cells) 0.0;
    c2 = Array.make (conv2 * cells) 0.0;
    feat = Array.make features 0.0;
    h = Array.make hidden 0.0;
    logits = Array.make actions 0.0;
  }

let action_of_id = function
  | 0 -> Game.Turn_left
  | 1 -> Game.Forward
  | _ -> Game.Turn_right

let split_ranges n parts =
  Array.init parts (fun i ->
      let a = i * n / parts in
      let b = (i + 1) * n / parts in
      (a, b))

let cell x y = (y * size) + x
let kindex dx dy = ((dy + 1) * kernel) + dx + 1
let patch_count = Array.make cells 0
let patch_cell = Array.make (cells * kernel_area) 0
let patch_kernel = Array.make (cells * kernel_area) 0

let () =
  for y = 0 to size - 1 do
    for x = 0 to size - 1 do
      let p = cell x y in
      for dy = -1 to 1 do
        let yy = y + dy in
        if yy >= 0 && yy < size then
          for dx = -1 to 1 do
            let xx = x + dx in
            if xx >= 0 && xx < size then (
              let i = (p * kernel_area) + patch_count.(p) in
              patch_cell.(i) <- cell xx yy;
              patch_kernel.(i) <- kindex dx dy;
              patch_count.(p) <- patch_count.(p) + 1)
          done
      done
    done
  done

let wc1_index f c k = (((f * channels) + c) * kernel_area) + k
let wc2_index g f k = (((g * conv1) + f) * kernel_area) + k
let wf_index j i = (j * features) + i
let wp_index a j = (a * hidden) + j

let init rng fan_in =
  let scale = sqrt (2.0 /. float_of_int fan_in) in
  (Rng.float rng *. 2.0 -. 1.0) *. scale

let create ~seed =
  let rng = Rng.create seed in
  {
    rng;
    wc1 = Array.init (conv1 * channels * kernel_area) (fun _ -> init rng (channels * kernel_area));
    bc1 = Array.make conv1 0.0;
    wc2 = Array.init (conv2 * conv1 * kernel_area) (fun _ -> init rng (conv1 * kernel_area));
    bc2 = Array.make conv2 0.0;
    wf = Array.init (hidden * features) (fun _ -> init rng features);
    bf = Array.make hidden 0.0;
    wp = Array.init (actions * hidden) (fun _ -> init rng hidden);
    bp = Array.make actions 0.0;
    wv = Array.init hidden (fun _ -> init rng hidden);
    bv = 0.0;
    mwc1 = Array.make (conv1 * channels * kernel_area) 0.0;
    vwc1 = Array.make (conv1 * channels * kernel_area) 0.0;
    mbc1 = Array.make conv1 0.0;
    vbc1 = Array.make conv1 0.0;
    mwc2 = Array.make (conv2 * conv1 * kernel_area) 0.0;
    vwc2 = Array.make (conv2 * conv1 * kernel_area) 0.0;
    mbc2 = Array.make conv2 0.0;
    vbc2 = Array.make conv2 0.0;
    mwf = Array.make (hidden * features) 0.0;
    vwf = Array.make (hidden * features) 0.0;
    mbf = Array.make hidden 0.0;
    vbf = Array.make hidden 0.0;
    mwp = Array.make (actions * hidden) 0.0;
    vwp = Array.make (actions * hidden) 0.0;
    mbp = Array.make actions 0.0;
    vbp = Array.make actions 0.0;
    mwv = Array.make hidden 0.0;
    vwv = Array.make hidden 0.0;
    mbv = 0.0;
    vbv = 0.0;
    adam_t = 0;
  }

let copy_model m =
  {
    rng = Rng.create 12345;
    wc1 = Array.copy m.wc1;
    bc1 = Array.copy m.bc1;
    wc2 = Array.copy m.wc2;
    bc2 = Array.copy m.bc2;
    wf = Array.copy m.wf;
    bf = Array.copy m.bf;
    wp = Array.copy m.wp;
    bp = Array.copy m.bp;
    wv = Array.copy m.wv;
    bv = m.bv;
    mwc1 = Array.copy m.mwc1;
    vwc1 = Array.copy m.vwc1;
    mbc1 = Array.copy m.mbc1;
    vbc1 = Array.copy m.vbc1;
    mwc2 = Array.copy m.mwc2;
    vwc2 = Array.copy m.vwc2;
    mbc2 = Array.copy m.mbc2;
    vbc2 = Array.copy m.vbc2;
    mwf = Array.copy m.mwf;
    vwf = Array.copy m.vwf;
    mbf = Array.copy m.mbf;
    vbf = Array.copy m.vbf;
    mwp = Array.copy m.mwp;
    vwp = Array.copy m.vwp;
    mbp = Array.copy m.mbp;
    vbp = Array.copy m.vbp;
    mwv = Array.copy m.mwv;
    vwv = Array.copy m.vwv;
    mbv = m.mbv;
    vbv = m.vbv;
    adam_t = m.adam_t;
  }

let fill_obs env obs off =
  for i = 0 to cells - 1 do
    let x = i mod size in
    let y = i / size in
    obs.(off + i) <-
      if x = env.Game.food_x && y = env.Game.food_y then 3
      else if x = env.Game.xs.(0) && y = env.Game.ys.(0) then 2
      else if env.Game.occupied.(i) then 1
      else 0
  done;
  obs.(off + cells) <-
    match env.Game.dir with Game.Up -> 0 | Game.Right -> 1 | Game.Down -> 2 | Game.Left -> 3

let forward m obs off w =
  for f = 0 to conv1 - 1 do
    for p = 0 to cells - 1 do
        let s = ref m.bc1.(f) in
        let base = p * kernel_area in
        for pi = 0 to patch_count.(p) - 1 do
          let q = patch_cell.(base + pi) in
          let c = obs.(off + q) in
          s := !s +. m.wc1.(wc1_index f c patch_kernel.(base + pi))
        done;
        w.c1.((f * cells) + p) <- tanh !s
    done
  done;
  for g = 0 to conv2 - 1 do
    for p = 0 to cells - 1 do
        let s = ref m.bc2.(g) in
        let base = p * kernel_area in
        for f = 0 to conv1 - 1 do
          for pi = 0 to patch_count.(p) - 1 do
            let q = patch_cell.(base + pi) in
            s :=
              !s +. (m.wc2.(wc2_index g f patch_kernel.(base + pi)) *. w.c1.((f * cells) + q))
          done
        done;
        w.c2.((g * cells) + p) <- tanh !s
    done
  done;
  for i = 0 to (conv2 * cells) - 1 do
    w.feat.(i) <- w.c2.(i)
  done;
  for i = (conv2 * cells) to features - 1 do
    w.feat.(i) <- 0.0
  done;
  w.feat.((conv2 * cells) + obs.(off + cells)) <- 1.0;
  dense_tanh_forward m.wf m.bf w.feat w.h hidden features;
  for a = 0 to actions - 1 do
    let s = ref m.bp.(a) in
    for j = 0 to hidden - 1 do
      s := !s +. (m.wp.(wp_index a j) *. w.h.(j))
    done;
    w.logits.(a) <- !s
  done;
  let v = ref m.bv in
  for j = 0 to hidden - 1 do
    v := !v +. (m.wv.(j) *. w.h.(j))
  done;
  !v

let probs logits =
  let mx = max logits.(0) (max logits.(1) logits.(2)) in
  let p0 = exp (logits.(0) -. mx) in
  let p1 = exp (logits.(1) -. mx) in
  let p2 = exp (logits.(2) -. mx) in
  let z = p0 +. p1 +. p2 in
  (p0 /. z, p1 /. z, p2 /. z)

let sample rng logits =
  let p0, p1, p2 = probs logits in
  let x = Rng.float rng in
  if x < p0 then (0, log p0)
  else if x < p0 +. p1 then (1, log p1)
  else (2, log p2)

let greedy m env =
  let obs = Array.make active 0 in
  let w = make_work () in
  fill_obs env obs 0;
  ignore (forward m obs 0 w);
  let best = ref 0 in
  for a = 1 to actions - 1 do
    if w.logits.(a) > w.logits.(!best) then best := a
  done;
  action_of_id !best

let collect ?(domains = 1) ~envs ~steps m =
  let env_count = Array.length envs in
  let n = env_count * steps in
  let r =
    {
      obs = Array.make (n * active) 0;
      actions = Array.make n 0;
      old_logp = Array.make n 0.0;
      rewards = Array.make n 0.0;
      dones = Array.make n false;
      values = Array.make n 0.0;
      last_values = Array.make env_count 0.0;
      adv = Array.make n 0.0;
      returns = Array.make n 0.0;
      n;
      envs = env_count;
    }
  in
  let domains = max 1 (min domains env_count) in
  let ranges = split_ranges env_count domains in
  let run_range (lo, hi) =
    let w = make_work () in
    for t = 0 to steps - 1 do
      for e = lo to hi - 1 do
        let k = (t * env_count) + e in
        let env = envs.(e) in
        let off = k * active in
        fill_obs env r.obs off;
        let v = forward m r.obs off w in
        let a, lp = sample env.Game.rng w.logits in
        let step = Game.step env (action_of_id a) in
        let idle = env.Game.since_food >= idle_limit env in
        r.actions.(k) <- a;
        r.old_logp.(k) <- lp;
        r.values.(k) <- v;
        r.rewards.(k) <- (if idle then -1.0 else step.Game.reward);
        r.dones.(k) <- step.done_ || idle;
        if step.done_ || idle then Game.reset env
      done
    done;
    for e = lo to hi - 1 do
      let tmp = Array.make active 0 in
      fill_obs envs.(e) tmp 0;
      r.last_values.(e) <- forward m tmp 0 w
    done
  in
  let handles =
    Array.init (domains - 1) (fun i -> Domain.spawn (fun () -> run_range ranges.(i + 1)))
  in
  run_range ranges.(0);
  Array.iter Domain.join handles;
  r

let compute_advantages ?(gamma = 0.99) ?(lambda = 0.95) r =
  for e = 0 to r.envs - 1 do
    let gae = ref 0.0 in
    for t = (r.n / r.envs) - 1 downto 0 do
      let k = (t * r.envs) + e in
      let next_value =
        if r.dones.(k) then 0.0
        else if t = (r.n / r.envs) - 1 then r.last_values.(e)
        else r.values.(k + r.envs)
      in
      let next_not_done = if r.dones.(k) then 0.0 else 1.0 in
      let delta = r.rewards.(k) +. (gamma *. next_value *. next_not_done) -. r.values.(k) in
      gae := delta +. (gamma *. lambda *. next_not_done *. !gae);
      r.adv.(k) <- !gae;
      r.returns.(k) <- !gae +. r.values.(k)
    done
  done;
  let mean = Array.fold_left ( +. ) 0.0 r.adv /. float_of_int r.n in
  let var =
    Array.fold_left
      (fun acc x ->
        let d = x -. mean in
        acc +. (d *. d))
      0.0 r.adv
    /. float_of_int r.n
  in
  let std = sqrt (var +. 1e-8) in
  for i = 0 to r.n - 1 do
    r.adv.(i) <- (r.adv.(i) -. mean) /. std
  done

let adam m p mt vt g i =
  mt.(i) <- (0.9 *. mt.(i)) +. (0.1 *. g);
  vt.(i) <- (0.999 *. vt.(i)) +. (0.001 *. g *. g);
  let t = float_of_int m.adam_t in
  let mh = mt.(i) /. (1.0 -. (0.9 ** t)) in
  let vh = vt.(i) /. (1.0 -. (0.999 ** t)) in
  p.(i) <- p.(i) -. (!lr *. mh /. (sqrt vh +. 1e-8))

let shuffle rng xs =
  for i = Array.length xs - 1 downto 1 do
    let j = Rng.int rng (i + 1) in
    let x = xs.(i) in
    xs.(i) <- xs.(j);
    xs.(j) <- x
  done

let update ?(epochs = 2) ?(batch = 512) ?(clip = 0.2) ?(vf_coef = 0.5) ?(ent_coef = 0.02) m r =
  let gwc1 = Array.make (Array.length m.wc1) 0.0 in
  let gbc1 = Array.make conv1 0.0 in
  let gwc2 = Array.make (Array.length m.wc2) 0.0 in
  let gbc2 = Array.make conv2 0.0 in
  let gwf = Array.make (Array.length m.wf) 0.0 in
  let gbf = Array.make hidden 0.0 in
  let gwp = Array.make (Array.length m.wp) 0.0 in
  let gbp = Array.make actions 0.0 in
  let gwv = Array.make hidden 0.0 in
  let gbv = ref 0.0 in
  let dlogits = Array.make actions 0.0 in
  let dh = Array.make hidden 0.0 in
  let dfeat = Array.make features 0.0 in
  let dc2 = Array.make (conv2 * cells) 0.0 in
  let dc1 = Array.make (conv1 * cells) 0.0 in
  let w = make_work () in
  let order = Array.init r.n (fun i -> i) in
  for _ = 1 to epochs do
    shuffle m.rng order;
    let start = ref 0 in
    while !start < r.n do
      let stop = min r.n (!start + batch) in
      let scale = 1.0 /. float_of_int (stop - !start) in
      Array.fill gwc1 0 (Array.length gwc1) 0.0;
      Array.fill gbc1 0 conv1 0.0;
      Array.fill gwc2 0 (Array.length gwc2) 0.0;
      Array.fill gbc2 0 conv2 0.0;
      Array.fill gwf 0 (Array.length gwf) 0.0;
      Array.fill gbf 0 hidden 0.0;
      Array.fill gwp 0 (Array.length gwp) 0.0;
      Array.fill gbp 0 actions 0.0;
      Array.fill gwv 0 hidden 0.0;
      gbv := 0.0;
      for n = !start to stop - 1 do
        let k = order.(n) in
        let off = k * active in
        let v = forward m r.obs off w in
        let p0, p1, p2 = probs w.logits in
        let ps = [| p0; p1; p2 |] in
        let a = r.actions.(k) in
        let lp = log ps.(a) in
        let ratio = exp (lp -. r.old_logp.(k)) in
        let adv = r.adv.(k) in
        let clipped =
          if adv >= 0.0 then min ratio (1.0 +. clip) else max ratio (1.0 -. clip)
        in
        let use_policy = (ratio *. adv) <= (clipped *. adv) in
        Array.fill dlogits 0 actions 0.0;
        if use_policy then
          for i = 0 to actions - 1 do
            let y = if i = a then 1.0 else 0.0 in
            dlogits.(i) <- dlogits.(i) -. (ratio *. adv *. (y -. ps.(i)) *. scale)
          done;
        let xlogx x = if x <= 0.0 then 0.0 else x *. log x in
        let entropy = -.((xlogx p0) +. (xlogx p1) +. (xlogx p2)) in
        for i = 0 to actions - 1 do
          dlogits.(i) <- dlogits.(i) +. (ent_coef *. ps.(i) *. (log ps.(i) +. entropy) *. scale)
        done;
        Array.fill dh 0 hidden 0.0;
        for i = 0 to actions - 1 do
          gbp.(i) <- gbp.(i) +. dlogits.(i);
          for j = 0 to hidden - 1 do
            let wi = wp_index i j in
            gwp.(wi) <- gwp.(wi) +. (dlogits.(i) *. w.h.(j));
            dh.(j) <- dh.(j) +. (dlogits.(i) *. m.wp.(wi))
          done
        done;
        let dv = vf_coef *. 2.0 *. (v -. r.returns.(k)) *. scale in
        gbv := !gbv +. dv;
        for j = 0 to hidden - 1 do
          gwv.(j) <- gwv.(j) +. (dv *. w.h.(j));
          dh.(j) <- dh.(j) +. (dv *. m.wv.(j))
        done;
        Array.fill dfeat 0 features 0.0;
        dense_tanh_backward m.wf w.feat w.h dh gwf gbf dfeat hidden features;
        Array.fill dc2 0 (conv2 * cells) 0.0;
        Array.fill dc1 0 (conv1 * cells) 0.0;
        for i = 0 to (conv2 * cells) - 1 do
          dc2.(i) <- dfeat.(i)
        done;
        for g = 0 to conv2 - 1 do
          for p = 0 to cells - 1 do
              let cp = (g * cells) + p in
              let dz = dc2.(cp) *. (1.0 -. (w.c2.(cp) *. w.c2.(cp))) in
              gbc2.(g) <- gbc2.(g) +. dz;
              let base = p * kernel_area in
              for f = 0 to conv1 - 1 do
                for pi = 0 to patch_count.(p) - 1 do
                  let q = patch_cell.(base + pi) in
                  let wi = wc2_index g f patch_kernel.(base + pi) in
                  gwc2.(wi) <- gwc2.(wi) +. (dz *. w.c1.((f * cells) + q));
                  dc1.((f * cells) + q) <- dc1.((f * cells) + q) +. (dz *. m.wc2.(wi))
                done
              done
          done
        done;
        for f = 0 to conv1 - 1 do
          for p = 0 to cells - 1 do
              let cp = (f * cells) + p in
              let dz = dc1.(cp) *. (1.0 -. (w.c1.(cp) *. w.c1.(cp))) in
              gbc1.(f) <- gbc1.(f) +. dz;
              let base = p * kernel_area in
              for pi = 0 to patch_count.(p) - 1 do
                let q = patch_cell.(base + pi) in
                let c = r.obs.(off + q) in
                let wi = wc1_index f c patch_kernel.(base + pi) in
                gwc1.(wi) <- gwc1.(wi) +. dz
              done
          done
        done
      done;
      m.adam_t <- m.adam_t + 1;
      for i = 0 to Array.length m.wc1 - 1 do
        adam m m.wc1 m.mwc1 m.vwc1 gwc1.(i) i
      done;
      for i = 0 to conv1 - 1 do
        adam m m.bc1 m.mbc1 m.vbc1 gbc1.(i) i
      done;
      for i = 0 to Array.length m.wc2 - 1 do
        adam m m.wc2 m.mwc2 m.vwc2 gwc2.(i) i
      done;
      for i = 0 to conv2 - 1 do
        adam m m.bc2 m.mbc2 m.vbc2 gbc2.(i) i
      done;
      for i = 0 to Array.length m.wf - 1 do
        adam m m.wf m.mwf m.vwf gwf.(i) i
      done;
      for i = 0 to hidden - 1 do
        adam m m.bf m.mbf m.vbf gbf.(i) i;
        adam m m.wv m.mwv m.vwv gwv.(i) i
      done;
      for i = 0 to Array.length m.wp - 1 do
        adam m m.wp m.mwp m.vwp gwp.(i) i
      done;
      for i = 0 to actions - 1 do
        adam m m.bp m.mbp m.vbp gbp.(i) i
      done;
      m.mbv <- (0.9 *. m.mbv) +. (0.1 *. !gbv);
      m.vbv <- (0.999 *. m.vbv) +. (0.001 *. !gbv *. !gbv);
      let t = float_of_int m.adam_t in
      let mh = m.mbv /. (1.0 -. (0.9 ** t)) in
      let vh = m.vbv /. (1.0 -. (0.999 ** t)) in
      m.bv <- m.bv -. (!lr *. mh /. (sqrt vh +. 1e-8));
      start := stop
    done
  done

let eval ?(episodes = 128) m =
  let total = ref 0 in
  let best = ref 0 in
  let total_steps = ref 0 in
  for i = 1 to episodes do
    let env = Game.create ~width:size ~height:size ~seed:(50_000 + i) in
    while Game.alive env && env.Game.since_food < idle_limit env do
      ignore (Game.step env (greedy m env))
    done;
    total := !total + Game.score env;
    best := max !best (Game.score env);
    total_steps := !total_steps + Game.steps env
  done;
  (float_of_int !total /. float_of_int episodes, !best, float_of_int !total_steps /. float_of_int episodes)

let train ?snapshot ?status ?stop ?iters ?(render_progress = true) ?(env_count = 64) ?(steps = 64)
    ?(learning_rate = 0.003) () =
  lr := learning_rate;
  let m = create ~seed:8 in
  let best = ref (copy_model m) in
  let best_score = ref neg_infinity in
  let envs = Array.init env_count (fun i -> Game.create ~width:size ~height:size ~seed:(1_000 + i)) in
  let t0 = Sys.time () in
  let should_stop () = match stop with Some stop -> Atomic.get stop | None -> false in
  let should_continue iter =
    (not (should_stop ())) && match iters with None -> true | Some total -> iter <= total
  in
  let publish iter msg =
    Option.iter (fun dst -> Atomic.set dst msg) status;
    if render_progress then
      match iters with
      | Some total -> Progress.draw ~current:iter ~total msg
      | None -> Printf.printf "\riter=%d %s%!" iter msg
  in
  let iter = ref 1 in
  while should_continue !iter do
    let r = collect ~envs ~steps m in
    compute_advantages r;
    update m r;
    let dt = max 1e-9 (Sys.time () -. t0) in
    let final_iter = match iters with Some total -> !iter = total | None -> false in
    if final_iter || !iter mod 25 = 0 then
      let mean, best_ep, _ = eval ~episodes:32 m in
      if mean > !best_score then (
        best_score := mean;
        best := copy_model m);
      publish !iter
        (Printf.sprintf "score=%.2f best=%d envs=%d %.0f sample/s" mean best_ep env_count
           (float_of_int (!iter * env_count * steps) /. dt))
    else
      publish !iter
        (Printf.sprintf "envs=%d %.0f sample/s" env_count (float_of_int (!iter * env_count * steps) /. dt));
    Option.iter (fun dst -> Atomic.set dst (Some (!iter, copy_model m))) snapshot;
    incr iter
  done;
  if render_progress then print_newline ();
  !best
