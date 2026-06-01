#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { SIZE = 8, CELLS = 64, CHANNELS = 4, ACTIONS = 3, CONV1 = 6, CONV2 = 6, K = 3, KA = 9 };
enum { DIRS = 4, FEATURES = CONV2 * CELLS + DIRS, HIDDEN = 96, ACTIVE = CELLS + 1 };

typedef struct { uint64_t s; } Rng;

typedef struct {
  int xs[CELLS], ys[CELLS], occ[CELLS];
  Rng rng;
  int len, dir, food_x, food_y, alive, score, steps, since_food;
} Env;

typedef struct {
  double wc1[CONV1 * CHANNELS * KA], bc1[CONV1], wc2[CONV2 * CONV1 * KA], bc2[CONV2];
  double wf[HIDDEN * FEATURES], bf[HIDDEN], wp[ACTIONS * HIDDEN], bp[ACTIONS], wv[HIDDEN], bv;
  double mwc1[CONV1 * CHANNELS * KA], vwc1[CONV1 * CHANNELS * KA], mbc1[CONV1], vbc1[CONV1];
  double mwc2[CONV2 * CONV1 * KA], vwc2[CONV2 * CONV1 * KA], mbc2[CONV2], vbc2[CONV2];
  double mwf[HIDDEN * FEATURES], vwf[HIDDEN * FEATURES], mbf[HIDDEN], vbf[HIDDEN];
  double mwp[ACTIONS * HIDDEN], vwp[ACTIONS * HIDDEN], mbp[ACTIONS], vbp[ACTIONS];
  double mwv[HIDDEN], vwv[HIDDEN], mbv, vbv;
  int adam_t;
} Model;

typedef struct {
  double c1[CONV1 * CELLS], c2[CONV2 * CELLS], feat[FEATURES], h[HIDDEN], logits[ACTIONS];
} Work;

typedef struct {
  int *obs, *actions, n, envs;
  double *old_logp, *rewards, *values, *last_values, *adv, *returns;
  unsigned char *dones;
} Rollout;

static int patch_count[CELLS], patch_cell[CELLS * KA], patch_kernel[CELLS * KA];
static double lr = 0.003;

static int cell(int x, int y) { return y * SIZE + x; }
static int kidx(int dx, int dy) { return (dy + 1) * K + dx + 1; }
static int wc1i(int f, int c, int k) { return ((f * CHANNELS + c) * KA) + k; }
static int wc2i(int g, int f, int k) { return ((g * CONV1 + f) * KA) + k; }
static int wpi(int a, int j) { return a * HIDDEN + j; }

static uint64_t mix_seed(int seed) {
  uint64_t x = (uint64_t)seed ^ 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static Rng rng_create(int seed) {
  Rng r = { mix_seed(seed) };
  if (!r.s) r.s = 0xD1B54A32D192ED03ULL;
  return r;
}

static uint64_t rng_next(Rng *r) {
  uint64_t x = r->s;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  r->s = x;
  return x;
}

static int rng_int(Rng *r, int bound) { return (int)((rng_next(r) & 0x7fffffffffffffffULL) % (uint64_t)bound); }
static double rng_float(Rng *r) { return (double)(rng_next(r) >> 11) / 9007199254740992.0; }

static void init_patches(void) {
  memset(patch_count, 0, sizeof patch_count);
  for (int y = 0; y < SIZE; y++) {
    for (int x = 0; x < SIZE; x++) {
      int p = cell(x, y);
      for (int dy = -1; dy <= 1; dy++) {
        int yy = y + dy;
        if (yy < 0 || yy >= SIZE) continue;
        for (int dx = -1; dx <= 1; dx++) {
          int xx = x + dx;
          if (xx < 0 || xx >= SIZE) continue;
          int i = p * KA + patch_count[p]++;
          patch_cell[i] = cell(xx, yy);
          patch_kernel[i] = kidx(dx, dy);
        }
      }
    }
  }
}

static void place_food(Env *e) {
  int free = CELLS - e->len;
  if (free <= 0) {
    e->food_x = e->food_y = -1;
    return;
  }
  int target = rng_int(&e->rng, free), seen = 0, chosen = 0;
  for (int i = 0; i < CELLS; i++) {
    if (!e->occ[i] && seen <= target) {
      chosen = i;
      seen++;
    }
  }
  e->food_x = chosen % SIZE;
  e->food_y = chosen / SIZE;
}

static void env_reset(Env *e) {
  memset(e->occ, 0, sizeof e->occ);
  int cx = SIZE / 2, cy = SIZE / 2;
  e->len = 3;
  e->dir = 1;
  e->alive = 1;
  e->score = e->steps = e->since_food = 0;
  for (int i = 0; i < e->len; i++) {
    e->xs[i] = cx - i;
    e->ys[i] = cy;
    e->occ[cell(e->xs[i], e->ys[i])] = 1;
  }
  place_food(e);
}

static Env env_create(int seed) {
  Env e;
  memset(&e, 0, sizeof e);
  e.rng = rng_create(seed);
  env_reset(&e);
  return e;
}

static void delta(int dir, int *dx, int *dy) {
  static const int dd[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
  *dx = dd[dir][0];
  *dy = dd[dir][1];
}

static int next_dir(int dir, int action) {
  if (action == 0) return (dir + 3) & 3;
  if (action == 2) return (dir + 1) & 3;
  return dir;
}

static int collides(Env *e, int x, int y, int grow) {
  if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return 1;
  int idx = cell(x, y);
  if (!e->occ[idx]) return 0;
  int tail = e->len - 1;
  return grow || x != e->xs[tail] || y != e->ys[tail];
}

static double env_step(Env *e, int action, int *done) {
  *done = 0;
  if (!e->alive) {
    *done = 1;
    return 0.0;
  }
  int nd = next_dir(e->dir, action), dx, dy;
  delta(nd, &dx, &dy);
  int nx = e->xs[0] + dx, ny = e->ys[0] + dy;
  int grow = nx == e->food_x && ny == e->food_y;
  e->steps++;
  if (collides(e, nx, ny, grow)) {
    e->alive = 0;
    *done = 1;
    return -1.0;
  }
  e->dir = nd;
  if (grow) e->len++;
  else e->occ[cell(e->xs[e->len - 1], e->ys[e->len - 1])] = 0;
  for (int i = e->len - 1; i >= 1; i--) {
    e->xs[i] = e->xs[i - 1];
    e->ys[i] = e->ys[i - 1];
  }
  e->xs[0] = nx;
  e->ys[0] = ny;
  e->occ[cell(nx, ny)] = 1;
  if (grow) {
    e->score++;
    e->since_food = 0;
    place_food(e);
    return 1.0;
  }
  e->since_food++;
  return -0.01;
}

static int idle_limit(Env *e) {
  int v = 24 + e->len;
  return v < CELLS ? v : CELLS;
}

static double init_weight(Rng *r, int fan_in) {
  double scale = sqrt(2.0 / (double)fan_in);
  return (rng_float(r) * 2.0 - 1.0) * scale;
}

static Model *model_create(int seed) {
  Model *m = calloc(1, sizeof(Model));
  if (!m) {
    perror("calloc model");
    exit(1);
  }
  Rng r = rng_create(seed);
  for (int i = 0; i < CONV1 * CHANNELS * KA; i++) m->wc1[i] = init_weight(&r, CHANNELS * KA);
  for (int i = 0; i < CONV2 * CONV1 * KA; i++) m->wc2[i] = init_weight(&r, CONV1 * KA);
  for (int i = 0; i < HIDDEN * FEATURES; i++) m->wf[i] = init_weight(&r, FEATURES);
  for (int i = 0; i < ACTIONS * HIDDEN; i++) m->wp[i] = init_weight(&r, HIDDEN);
  for (int i = 0; i < HIDDEN; i++) m->wv[i] = init_weight(&r, HIDDEN);
  return m;
}

static Model *model_copy(Model *src) {
  Model *dst = malloc(sizeof(Model));
  if (!dst) {
    perror("malloc model copy");
    exit(1);
  }
  memcpy(dst, src, sizeof(Model));
  return dst;
}

static void fill_obs(Env *e, int *obs) {
  for (int i = 0; i < CELLS; i++) {
    int x = i % SIZE, y = i / SIZE;
    obs[i] = (x == e->food_x && y == e->food_y) ? 3 : (x == e->xs[0] && y == e->ys[0]) ? 2 : e->occ[i] ? 1 : 0;
  }
  obs[CELLS] = e->dir;
}

static double forward(Model *m, const int *obs, Work *w) {
  for (int f = 0; f < CONV1; f++) {
    for (int p = 0; p < CELLS; p++) {
      double s = m->bc1[f];
      int base = p * KA;
      for (int pi = 0; pi < patch_count[p]; pi++) {
        s += m->wc1[wc1i(f, obs[patch_cell[base + pi]], patch_kernel[base + pi])];
      }
      w->c1[f * CELLS + p] = tanh(s);
    }
  }
  for (int g = 0; g < CONV2; g++) {
    for (int p = 0; p < CELLS; p++) {
      double s = m->bc2[g];
      int base = p * KA;
      for (int f = 0; f < CONV1; f++) {
        double *c1 = &w->c1[f * CELLS];
        for (int pi = 0; pi < patch_count[p]; pi++) {
          int q = patch_cell[base + pi];
          s += m->wc2[wc2i(g, f, patch_kernel[base + pi])] * c1[q];
        }
      }
      w->c2[g * CELLS + p] = tanh(s);
    }
  }
  memcpy(w->feat, w->c2, sizeof(double) * CONV2 * CELLS);
  memset(w->feat + CONV2 * CELLS, 0, sizeof(double) * DIRS);
  w->feat[CONV2 * CELLS + obs[CELLS]] = 1.0;
  for (int j = 0; j < HIDDEN; j++) {
    double s = m->bf[j], *wf = &m->wf[j * FEATURES];
    for (int i = 0; i < FEATURES; i++) s += wf[i] * w->feat[i];
    w->h[j] = tanh(s);
  }
  for (int a = 0; a < ACTIONS; a++) {
    double s = m->bp[a], *wp = &m->wp[a * HIDDEN];
    for (int j = 0; j < HIDDEN; j++) s += wp[j] * w->h[j];
    w->logits[a] = s;
  }
  double v = m->bv;
  for (int j = 0; j < HIDDEN; j++) v += m->wv[j] * w->h[j];
  return v;
}

static void probs(const double *logits, double *p) {
  double mx = logits[0] > logits[1] ? logits[0] : logits[1];
  if (logits[2] > mx) mx = logits[2];
  double z = 0.0;
  for (int i = 0; i < ACTIONS; i++) {
    p[i] = exp(logits[i] - mx);
    z += p[i];
  }
  for (int i = 0; i < ACTIONS; i++) p[i] /= z;
}

static int sample_action(Rng *r, const double *logits, double *lp) {
  double p[ACTIONS];
  probs(logits, p);
  double x = rng_float(r);
  int a = x < p[0] ? 0 : x < p[0] + p[1] ? 1 : 2;
  *lp = log(p[a]);
  return a;
}

static Rollout rollout_alloc(int envs, int steps) {
  int n = envs * steps;
  Rollout r;
  r.n = n;
  r.envs = envs;
  r.obs = calloc((size_t)n * ACTIVE, sizeof(int));
  r.actions = calloc(n, sizeof(int));
  r.old_logp = calloc(n, sizeof(double));
  r.rewards = calloc(n, sizeof(double));
  r.dones = calloc(n, 1);
  r.values = calloc(n, sizeof(double));
  r.last_values = calloc(envs, sizeof(double));
  r.adv = calloc(n, sizeof(double));
  r.returns = calloc(n, sizeof(double));
  if (!r.obs || !r.actions || !r.old_logp || !r.rewards || !r.dones || !r.values || !r.last_values || !r.adv || !r.returns) {
    perror("calloc rollout");
    exit(1);
  }
  return r;
}

static void rollout_free(Rollout *r) {
  free(r->obs);
  free(r->actions);
  free(r->old_logp);
  free(r->rewards);
  free(r->dones);
  free(r->values);
  free(r->last_values);
  free(r->adv);
  free(r->returns);
}

static Rollout collect(Model *m, Env *envs, int env_count, int steps) {
  Rollout r = rollout_alloc(env_count, steps);
  Work w;
  for (int t = 0; t < steps; t++) {
    for (int e = 0; e < env_count; e++) {
      int k = t * env_count + e;
      Env *env = &envs[e];
      int *obs = &r.obs[k * ACTIVE];
      fill_obs(env, obs);
      r.values[k] = forward(m, obs, &w);
      double lp;
      int a = sample_action(&env->rng, w.logits, &lp);
      int done;
      double reward = env_step(env, a, &done);
      int idle = env->since_food >= idle_limit(env);
      r.actions[k] = a;
      r.old_logp[k] = lp;
      r.rewards[k] = idle ? -1.0 : reward;
      r.dones[k] = (unsigned char)(done || idle);
      if (done || idle) env_reset(env);
    }
  }
  for (int e = 0; e < env_count; e++) {
    int obs[ACTIVE];
    fill_obs(&envs[e], obs);
    r.last_values[e] = forward(m, obs, &w);
  }
  return r;
}

static void advantages(Rollout *r) {
  int steps = r->n / r->envs;
  for (int e = 0; e < r->envs; e++) {
    double gae = 0.0;
    for (int t = steps - 1; t >= 0; t--) {
      int k = t * r->envs + e;
      double next_value = r->dones[k] ? 0.0 : (t == steps - 1 ? r->last_values[e] : r->values[k + r->envs]);
      double nd = r->dones[k] ? 0.0 : 1.0;
      double delta = r->rewards[k] + 0.99 * next_value * nd - r->values[k];
      gae = delta + 0.99 * 0.95 * nd * gae;
      r->adv[k] = gae;
      r->returns[k] = gae + r->values[k];
    }
  }
  double mean = 0.0, var = 0.0;
  for (int i = 0; i < r->n; i++) mean += r->adv[i];
  mean /= (double)r->n;
  for (int i = 0; i < r->n; i++) {
    double d = r->adv[i] - mean;
    var += d * d;
  }
  double std = sqrt(var / (double)r->n + 1e-8);
  for (int i = 0; i < r->n; i++) r->adv[i] = (r->adv[i] - mean) / std;
}

static void shuffle(Rng *rng, int *xs, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = rng_int(rng, i + 1);
    int x = xs[i];
    xs[i] = xs[j];
    xs[j] = x;
  }
}

static void adam(double *p, double *m, double *v, double g, int i, int t) {
  m[i] = 0.9 * m[i] + 0.1 * g;
  v[i] = 0.999 * v[i] + 0.001 * g * g;
  double mh = m[i] / (1.0 - pow(0.9, t));
  double vh = v[i] / (1.0 - pow(0.999, t));
  p[i] -= lr * mh / (sqrt(vh) + 1e-8);
}

static void zero(double *x, int n) { memset(x, 0, (size_t)n * sizeof(double)); }

static void update(Model *m, Rollout *r, Rng *rng) {
  static double gwc1[CONV1 * CHANNELS * KA], gbc1[CONV1], gwc2[CONV2 * CONV1 * KA], gbc2[CONV2];
  static double gwf[HIDDEN * FEATURES], gbf[HIDDEN], gwp[ACTIONS * HIDDEN], gbp[ACTIONS], gwv[HIDDEN];
  static double dlogits[ACTIONS], dh[HIDDEN], dfeat[FEATURES], dc2[CONV2 * CELLS], dc1[CONV1 * CELLS];
  int *order = malloc((size_t)r->n * sizeof(int));
  if (!order) {
    perror("malloc order");
    exit(1);
  }
  for (int i = 0; i < r->n; i++) order[i] = i;
  Work w;
  for (int epoch = 0; epoch < 2; epoch++) {
    shuffle(rng, order, r->n);
    for (int start = 0; start < r->n; start += 512) {
      int stop = start + 512 < r->n ? start + 512 : r->n;
      double scale = 1.0 / (double)(stop - start), gbv = 0.0;
      zero(gwc1, CONV1 * CHANNELS * KA);
      zero(gbc1, CONV1);
      zero(gwc2, CONV2 * CONV1 * KA);
      zero(gbc2, CONV2);
      zero(gwf, HIDDEN * FEATURES);
      zero(gbf, HIDDEN);
      zero(gwp, ACTIONS * HIDDEN);
      zero(gbp, ACTIONS);
      zero(gwv, HIDDEN);
      for (int nn = start; nn < stop; nn++) {
        int k = order[nn], *obs = &r->obs[k * ACTIVE];
        double v = forward(m, obs, &w), p[ACTIONS];
        probs(w.logits, p);
        int a = r->actions[k];
        double ratio = exp(log(p[a]) - r->old_logp[k]);
        double adv = r->adv[k];
        double clipped = adv >= 0.0 ? fmin(ratio, 1.2) : fmax(ratio, 0.8);
        int use_policy = ratio * adv <= clipped * adv;
        memset(dlogits, 0, sizeof dlogits);
        if (use_policy) {
          for (int i = 0; i < ACTIONS; i++) {
            double y = i == a ? 1.0 : 0.0;
            dlogits[i] -= ratio * adv * (y - p[i]) * scale;
          }
        }
        double entropy = 0.0;
        for (int i = 0; i < ACTIONS; i++) entropy -= p[i] <= 0.0 ? 0.0 : p[i] * log(p[i]);
        for (int i = 0; i < ACTIONS; i++) dlogits[i] += 0.02 * p[i] * (log(p[i]) + entropy) * scale;
        memset(dh, 0, sizeof dh);
        for (int i = 0; i < ACTIONS; i++) {
          gbp[i] += dlogits[i];
          for (int j = 0; j < HIDDEN; j++) {
            int wi = wpi(i, j);
            gwp[wi] += dlogits[i] * w.h[j];
            dh[j] += dlogits[i] * m->wp[wi];
          }
        }
        double dv = (v - r->returns[k]) * scale;
        gbv += dv;
        for (int j = 0; j < HIDDEN; j++) {
          gwv[j] += dv * w.h[j];
          dh[j] += dv * m->wv[j];
        }
        zero(dfeat, FEATURES);
        for (int j = 0; j < HIDDEN; j++) {
          double dz = dh[j] * (1.0 - w.h[j] * w.h[j]);
          gbf[j] += dz;
          double *gwfj = &gwf[j * FEATURES], *wfj = &m->wf[j * FEATURES];
          for (int i = 0; i < FEATURES; i++) {
            gwfj[i] += dz * w.feat[i];
            dfeat[i] += dz * wfj[i];
          }
        }
        zero(dc2, CONV2 * CELLS);
        zero(dc1, CONV1 * CELLS);
        memcpy(dc2, dfeat, sizeof(double) * CONV2 * CELLS);
        for (int g = 0; g < CONV2; g++) {
          for (int pp = 0; pp < CELLS; pp++) {
            int cp = g * CELLS + pp;
            double dz = dc2[cp] * (1.0 - w.c2[cp] * w.c2[cp]);
            gbc2[g] += dz;
            int base = pp * KA;
            for (int f = 0; f < CONV1; f++) {
              for (int pi = 0; pi < patch_count[pp]; pi++) {
                int q = patch_cell[base + pi], wi = wc2i(g, f, patch_kernel[base + pi]);
                gwc2[wi] += dz * w.c1[f * CELLS + q];
                dc1[f * CELLS + q] += dz * m->wc2[wi];
              }
            }
          }
        }
        for (int f = 0; f < CONV1; f++) {
          for (int pp = 0; pp < CELLS; pp++) {
            int cp = f * CELLS + pp;
            double dz = dc1[cp] * (1.0 - w.c1[cp] * w.c1[cp]);
            gbc1[f] += dz;
            int base = pp * KA;
            for (int pi = 0; pi < patch_count[pp]; pi++) {
              int q = patch_cell[base + pi];
              gwc1[wc1i(f, obs[q], patch_kernel[base + pi])] += dz;
            }
          }
        }
      }
      m->adam_t++;
      for (int i = 0; i < CONV1 * CHANNELS * KA; i++) adam(m->wc1, m->mwc1, m->vwc1, gwc1[i], i, m->adam_t);
      for (int i = 0; i < CONV1; i++) adam(m->bc1, m->mbc1, m->vbc1, gbc1[i], i, m->adam_t);
      for (int i = 0; i < CONV2 * CONV1 * KA; i++) adam(m->wc2, m->mwc2, m->vwc2, gwc2[i], i, m->adam_t);
      for (int i = 0; i < CONV2; i++) adam(m->bc2, m->mbc2, m->vbc2, gbc2[i], i, m->adam_t);
      for (int i = 0; i < HIDDEN * FEATURES; i++) adam(m->wf, m->mwf, m->vwf, gwf[i], i, m->adam_t);
      for (int i = 0; i < HIDDEN; i++) {
        adam(m->bf, m->mbf, m->vbf, gbf[i], i, m->adam_t);
        adam(m->wv, m->mwv, m->vwv, gwv[i], i, m->adam_t);
      }
      for (int i = 0; i < ACTIONS * HIDDEN; i++) adam(m->wp, m->mwp, m->vwp, gwp[i], i, m->adam_t);
      for (int i = 0; i < ACTIONS; i++) adam(m->bp, m->mbp, m->vbp, gbp[i], i, m->adam_t);
      m->mbv = 0.9 * m->mbv + 0.1 * gbv;
      m->vbv = 0.999 * m->vbv + 0.001 * gbv * gbv;
      double mh = m->mbv / (1.0 - pow(0.9, m->adam_t));
      double vh = m->vbv / (1.0 - pow(0.999, m->adam_t));
      m->bv -= lr * mh / (sqrt(vh) + 1e-8);
    }
  }
  free(order);
}

static int greedy(Model *m, Env *e) {
  int obs[ACTIVE];
  Work w;
  fill_obs(e, obs);
  (void)forward(m, obs, &w);
  int best = 0;
  for (int a = 1; a < ACTIONS; a++)
    if (w.logits[a] > w.logits[best]) best = a;
  return best;
}

static void render_env(Env *e, char *buf, size_t n) {
  size_t pos = 0;
  for (int y = 0; y < SIZE && pos + SIZE + 1 < n; y++) {
    for (int x = 0; x < SIZE; x++) buf[pos++] = '.';
    buf[pos++] = '\n';
  }
  if (e->food_x >= 0) buf[e->food_y * (SIZE + 1) + e->food_x] = '*';
  for (int i = e->len - 1; i >= 0; i--) {
    buf[e->ys[i] * (SIZE + 1) + e->xs[i]] = i == 0 ? '#' : 'o';
  }
  buf[pos] = 0;
}

static void preview(Model *m, int policy, const char *status, int seed) {
  Env e = env_create(seed);
  char board[(SIZE + 1) * SIZE + 1];
  while (e.alive && e.since_food < idle_limit(&e)) {
    render_env(&e, board, sizeof board);
    printf("\033[2J\033[Hsnake c ppo 8x8\npolicy=%d %s\nrollout score=%d steps=%d\n\n%s", policy, status, e.score,
           e.steps, board);
    fflush(stdout);
    int done;
    (void)env_step(&e, greedy(m, &e), &done);
    usleep(30000);
  }
}

static void eval_model(Model *m, double *mean, int *best, double *mean_steps) {
  int total = 0, total_steps = 0;
  *best = 0;
  for (int i = 1; i <= 128; i++) {
    Env e = env_create(50000 + i);
    while (e.alive && e.since_food < idle_limit(&e)) {
      int done;
      (void)env_step(&e, greedy(m, &e), &done);
    }
    total += e.score;
    if (e.score > *best) *best = e.score;
    total_steps += e.steps;
  }
  *mean = (double)total / 128.0;
  *mean_steps = (double)total_steps / 128.0;
}

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void train(int iters, int env_count, int show_preview) {
  Model *m = model_create(8);
  Model *pm = model_copy(m);
  Env *envs = calloc((size_t)env_count, sizeof(Env));
  if (!envs) {
    perror("calloc envs");
    exit(1);
  }
  for (int i = 0; i < env_count; i++) envs[i] = env_create(1000 + i);
  Rng rng = rng_create(99);
  double t0 = now_sec();
  for (int iter = 1; iter <= iters; iter++) {
    Rollout r = collect(m, envs, env_count, 64);
    advantages(&r);
    update(m, &r, &rng);
    rollout_free(&r);
    double dt = now_sec() - t0;
    char status[128];
    snprintf(status, sizeof status, "envs=%d %.0f sample/s", env_count, (double)(iter * env_count * 64) / dt);
    if (show_preview) {
      memcpy(pm, m, sizeof(Model));
      preview(pm, iter, status, 50000 + iter);
    } else {
      printf("\riter=%d %s", iter, status);
      fflush(stdout);
    }
  }
  double mean, mean_steps;
  int best;
  eval_model(m, &mean, &best, &mean_steps);
  printf("\nepisodes=128 mean_score=%.2f best=%d mean_steps=%.0f train_seconds=%.3f\n", mean, best, mean_steps, now_sec() - t0);
  free(envs);
  free(pm);
  free(m);
}

static void rollout_bench(int env_count, int steps, int iters) {
  Model *m = model_create(8);
  Env *envs = calloc((size_t)env_count, sizeof(Env));
  if (!envs) {
    perror("calloc envs");
    exit(1);
  }
  for (int i = 0; i < env_count; i++) envs[i] = env_create(1000 + i);
  double t0 = now_sec();
  for (int i = 0; i < iters; i++) {
    Rollout r = collect(m, envs, env_count, steps);
    rollout_free(&r);
  }
  double dt = now_sec() - t0, n = (double)env_count * steps * iters;
  printf("rollout envs=%d steps=%d iters=%d samples=%.0f seconds=%.3f samples_per_second=%.0f\n", env_count, steps, iters, n, dt, n / dt);
  free(envs);
  free(m);
}

int main(int argc, char **argv) {
  init_patches();
  if (argc >= 2 && strcmp(argv[1], "train") == 0) {
    train(argc >= 3 ? atoi(argv[2]) : 50, argc >= 4 ? atoi(argv[3]) : 64, 1);
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "bench-train") == 0) {
    train(argc >= 3 ? atoi(argv[2]) : 50, argc >= 4 ? atoi(argv[3]) : 64, 0);
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "rollout-bench") == 0) {
    rollout_bench(argc >= 3 ? atoi(argv[2]) : 256, argc >= 4 ? atoi(argv[3]) : 64, argc >= 5 ? atoi(argv[4]) : 20);
    return 0;
  }
  fprintf(stderr, "usage: %s {train [iters envs]|bench-train [iters envs]|rollout-bench [envs steps iters]}\n", argv[0]);
  return 2;
}
