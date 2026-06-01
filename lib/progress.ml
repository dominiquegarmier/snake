let draw ?(width = 32) ~current ~total msg =
  let total = max 1 total in
  let current = min total (max 0 current) in
  let filled = current * width / total in
  print_char '\r';
  print_char '[';
  for i = 1 to width do
    print_char (if i <= filled then '#' else '-')
  done;
  Printf.printf "] %3d%% %s%!" (current * 100 / total) msg

let finish ?(width = 32) msg =
  draw ~width ~current:1 ~total:1 msg;
  print_newline ()
