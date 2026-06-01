type t = { mutable state : int64 }

let mix_seed seed =
  let x = Int64.of_int seed in
  let x = Int64.logxor x 0x9E3779B97F4A7C15L in
  let x = Int64.mul (Int64.logxor x (Int64.shift_right_logical x 30)) 0xBF58476D1CE4E5B9L in
  let x = Int64.mul (Int64.logxor x (Int64.shift_right_logical x 27)) 0x94D049BB133111EBL in
  Int64.logxor x (Int64.shift_right_logical x 31)

let create seed =
  let state = mix_seed seed in
  { state = if state = 0L then 0xD1B54A32D192ED03L else state }

let next_u64 rng =
  let x = rng.state in
  let x = Int64.logxor x (Int64.shift_left x 13) in
  let x = Int64.logxor x (Int64.shift_right_logical x 7) in
  let x = Int64.logxor x (Int64.shift_left x 17) in
  rng.state <- x;
  x

let int rng bound =
  if bound <= 0 then invalid_arg "Rng.int: bound must be positive";
  let x = Int64.logand (next_u64 rng) Int64.max_int in
  Int64.to_int (Int64.rem x (Int64.of_int bound))

let float rng =
  let x = Int64.shift_right_logical (next_u64 rng) 11 in
  Int64.to_float x /. 9007199254740992.0
