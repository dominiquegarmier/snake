#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <math.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static double dot_f64(value w, value x, int off, int n)
{
  int i = 0;
  double sum = 0.0;
#if defined(__aarch64__)
  float64x2_t acc = vdupq_n_f64(0.0);
  for (; i + 1 < n; i += 2) {
    double wa[2] = { Double_field(w, off + i), Double_field(w, off + i + 1) };
    double xa[2] = { Double_field(x, i), Double_field(x, i + 1) };
    acc = vfmaq_f64(acc, vld1q_f64(wa), vld1q_f64(xa));
  }
  sum = vaddvq_f64(acc);
#endif
  for (; i < n; i++) {
    sum += Double_field(w, off + i) * Double_field(x, i);
  }
  return sum;
}

CAMLprim value snake_dense_tanh_forward(value w, value b, value x, value y, value rows_v, value cols_v)
{
  CAMLparam5(w, b, x, y, rows_v);
  CAMLxparam1(cols_v);
  int rows = Long_val(rows_v);
  int cols = Long_val(cols_v);
  for (int r = 0; r < rows; r++) {
    double s = Double_field(b, r) + dot_f64(w, x, r * cols, cols);
    Store_double_field(y, r, tanh(s));
  }
  CAMLreturn(Val_unit);
}

CAMLprim value snake_dense_tanh_forward_byte(value *argv, int argn)
{
  (void)argn;
  return snake_dense_tanh_forward(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value snake_dense_tanh_backward(value w, value x, value y, value dy, value gw, value gb,
                                         value dx, value rows_v, value cols_v)
{
  CAMLparam5(w, x, y, dy, gw);
  CAMLxparam4(gb, dx, rows_v, cols_v);
  int rows = Long_val(rows_v);
  int cols = Long_val(cols_v);
  for (int r = 0; r < rows; r++) {
    double yr = Double_field(y, r);
    double dz = Double_field(dy, r) * (1.0 - (yr * yr));
    Store_double_field(gb, r, Double_field(gb, r) + dz);
    int off = r * cols;
    int i = 0;
#if defined(__aarch64__)
    float64x2_t dzv = vdupq_n_f64(dz);
    for (; i + 1 < cols; i += 2) {
      double xv_a[2] = { Double_field(x, i), Double_field(x, i + 1) };
      double wv_a[2] = { Double_field(w, off + i), Double_field(w, off + i + 1) };
      double gw_a[2] = { Double_field(gw, off + i), Double_field(gw, off + i + 1) };
      double dx_a[2] = { Double_field(dx, i), Double_field(dx, i + 1) };
      float64x2_t xv = vld1q_f64(xv_a);
      float64x2_t wv = vld1q_f64(wv_a);
      float64x2_t gwv = vld1q_f64(gw_a);
      float64x2_t dxv = vld1q_f64(dx_a);
      gwv = vfmaq_f64(gwv, dzv, xv);
      dxv = vfmaq_f64(dxv, dzv, wv);
      vst1q_f64(gw_a, gwv);
      vst1q_f64(dx_a, dxv);
      Store_double_field(gw, off + i, gw_a[0]);
      Store_double_field(gw, off + i + 1, gw_a[1]);
      Store_double_field(dx, i, dx_a[0]);
      Store_double_field(dx, i + 1, dx_a[1]);
    }
#endif
    for (; i < cols; i++) {
      Store_double_field(gw, off + i, Double_field(gw, off + i) + (dz * Double_field(x, i)));
      Store_double_field(dx, i, Double_field(dx, i) + (dz * Double_field(w, off + i)));
    }
  }
  CAMLreturn(Val_unit);
}

CAMLprim value snake_dense_tanh_backward_byte(value *argv, int argn)
{
  (void)argn;
  return snake_dense_tanh_backward(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],
                                   argv[7], argv[8]);
}
