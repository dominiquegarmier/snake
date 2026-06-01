type direction = Up | Right | Down | Left
type action = Turn_left | Forward | Turn_right

type t = {
  width : int;
  height : int;
  xs : int array;
  ys : int array;
  occupied : bool array;
  rng : Rng.t;
  mutable len : int;
  mutable dir : direction;
  mutable food_x : int;
  mutable food_y : int;
  mutable alive : bool;
  mutable score : int;
  mutable steps : int;
  mutable since_food : int;
}

type step_result = {
  reward : float;
  done_ : bool;
  ate_food : bool;
}

let width t = t.width
let height t = t.height
let score t = t.score
let steps t = t.steps
let alive t = t.alive

let index t x y = (y * t.width) + x

let turn_left = function
  | Up -> Left
  | Left -> Down
  | Down -> Right
  | Right -> Up

let turn_right = function
  | Up -> Right
  | Right -> Down
  | Down -> Left
  | Left -> Up

let delta = function
  | Up -> (0, -1)
  | Right -> (1, 0)
  | Down -> (0, 1)
  | Left -> (-1, 0)

let dir_after_action dir = function
  | Turn_left -> turn_left dir
  | Forward -> dir
  | Turn_right -> turn_right dir

let place_food t =
  let free = (t.width * t.height) - t.len in
  if free <= 0 then (
    t.food_x <- -1;
    t.food_y <- -1)
  else (
    let target = Rng.int t.rng free in
    let seen = ref 0 in
    let chosen = ref 0 in
    for i = 0 to Array.length t.occupied - 1 do
      if (not t.occupied.(i)) && !seen <= target then (
        chosen := i;
        incr seen)
    done;
    t.food_x <- !chosen mod t.width;
    t.food_y <- !chosen / t.width)

let reset t =
  Array.fill t.occupied 0 (Array.length t.occupied) false;
  let cx = t.width / 2 in
  let cy = t.height / 2 in
  t.len <- min 3 t.width;
  t.dir <- Right;
  t.alive <- true;
  t.score <- 0;
  t.steps <- 0;
  t.since_food <- 0;
  for i = 0 to t.len - 1 do
    t.xs.(i) <- cx - i;
    t.ys.(i) <- cy;
    t.occupied.(index t t.xs.(i) t.ys.(i)) <- true
  done;
  place_food t

let create ~width ~height ~seed =
  if width < 4 || height < 4 then invalid_arg "Snake.create: board must be at least 4x4";
  let cells = width * height in
  let t =
    {
      width;
      height;
      xs = Array.make cells 0;
      ys = Array.make cells 0;
      occupied = Array.make cells false;
      rng = Rng.create seed;
      len = 0;
      dir = Right;
      food_x = 0;
      food_y = 0;
      alive = true;
      score = 0;
      steps = 0;
      since_food = 0;
    }
  in
  reset t;
  t

let collides t x y grow =
  if x < 0 || x >= t.width || y < 0 || y >= t.height then true
  else
    let idx = index t x y in
    if not t.occupied.(idx) then false
    else
      let tail = t.len - 1 in
      grow || x <> t.xs.(tail) || y <> t.ys.(tail)

let step t action =
  if not t.alive then { reward = 0.0; done_ = true; ate_food = false }
  else
    let next_dir = dir_after_action t.dir action in
    let dx, dy = delta next_dir in
    let nx = t.xs.(0) + dx in
    let ny = t.ys.(0) + dy in
    let grow = nx = t.food_x && ny = t.food_y in
    t.steps <- t.steps + 1;
    if collides t nx ny grow then (
      t.alive <- false;
      { reward = -1.0; done_ = true; ate_food = false })
    else (
      t.dir <- next_dir;
      if grow then t.len <- t.len + 1
      else (
        let tail = t.len - 1 in
        t.occupied.(index t t.xs.(tail) t.ys.(tail)) <- false);
      for i = t.len - 1 downto 1 do
        t.xs.(i) <- t.xs.(i - 1);
        t.ys.(i) <- t.ys.(i - 1)
      done;
      t.xs.(0) <- nx;
      t.ys.(0) <- ny;
      t.occupied.(index t nx ny) <- true;
      if grow then (
        t.score <- t.score + 1;
        t.since_food <- 0;
        place_food t;
        { reward = 1.0; done_ = false; ate_food = true })
      else (
        t.since_food <- t.since_food + 1;
        if t.since_food > t.width * t.height * 4 then (
          t.alive <- false;
          { reward = -1.0; done_ = true; ate_food = false })
        else { reward = -0.01; done_ = false; ate_food = false }))

let dangerous t dir =
  let dx, dy = delta dir in
  let nx = t.xs.(0) + dx in
  let ny = t.ys.(0) + dy in
  collides t nx ny false

let observation_size = 11

let observation t =
  let left = turn_left t.dir in
  let right = turn_right t.dir in
  [|
    (if dangerous t t.dir then 1.0 else 0.0);
    (if dangerous t right then 1.0 else 0.0);
    (if dangerous t left then 1.0 else 0.0);
    (if t.dir = Up then 1.0 else 0.0);
    (if t.dir = Right then 1.0 else 0.0);
    (if t.dir = Down then 1.0 else 0.0);
    (if t.dir = Left then 1.0 else 0.0);
    (float_of_int (t.food_x - t.xs.(0)) /. float_of_int t.width);
    (float_of_int (t.food_y - t.ys.(0)) /. float_of_int t.height);
    (float_of_int t.len /. float_of_int (t.width * t.height));
    (if t.alive then 1.0 else 0.0);
  |]

let render t =
  let stride = t.width + 1 in
  let buf = Bytes.make ((t.height * stride) + 1) ' ' in
  for y = 0 to t.height - 1 do
    for x = 0 to t.width - 1 do
      Bytes.set buf ((y * stride) + x) '.'
    done;
    Bytes.set buf ((y * stride) + t.width) '\n'
  done;
  if t.food_x >= 0 then Bytes.set buf ((t.food_y * stride) + t.food_x) '*';
  for i = t.len - 1 downto 0 do
    let c = if i = 0 then '#' else 'o' in
    Bytes.set buf ((t.ys.(i) * stride) + t.xs.(i)) c
  done;
  Bytes.set buf (Bytes.length buf - 1) '\n';
  Bytes.unsafe_to_string buf
