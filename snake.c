#include <math.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
  SIZE = 8,
  CELLS = 64,
  ACTIVE = CELLS + 1,
  CATS = 3,
  DIRS = 4,
  IN = CATS * CELLS + DIRS,
  H1 = 128,
  H2 = 64,
  ACTIONS = 3,
  ROLLOUT_STEPS = 32,
  EPOCHS = 3,
  BATCH = 1024
};

static const double VIZ_DT = 0.016;
static volatile sig_atomic_t stop = 0;

typedef struct { uint64_t s; } Rng;

typedef struct {
  int xs[CELLS], ys[CELLS];
  unsigned char occ[CELLS];
  uint64_t visited;
  Rng rng;
  int len, max_len, dir, food, alive, score, steps, since_food;
} Env;

typedef struct {
  float w1[H1 * IN], b1[H1], w2[H2 * H1], b2[H2], wp[ACTIONS * H2], bp[ACTIONS], wv[H2], bv;
  float mw1[H1 * IN], vw1[H1 * IN], mb1[H1], vb1[H1];
  float mw2[H2 * H1], vw2[H2 * H1], mb2[H2], vb2[H2];
  float mwp[ACTIONS * H2], vwp[ACTIONS * H2], mbp[ACTIONS], vbp[ACTIONS];
  float mwv[H2], vwv[H2], mbv, vbv;
  float b1pow, b2pow;
} Model;

typedef struct {
  float h1[H1], h2[H2], logits[ACTIONS];
  int feat[CELLS + DIRS], feat_n;
} Work;

typedef struct {
  unsigned char *obs, *actions, *dones;
  float *old_logp, *rewards, *values, *last_values, *adv, *returns;
  int n, envs;
} Rollout;

typedef struct {
  float mean_score, mean_peak_len, mean_len_cov, mean_visit_cov, p30_len, p30_visit, mean_steps;
  int best_score, best_len;
} Eval;

static float lr = 0.0007f;

static int cell(int x, int y) { return y * SIZE + x; }

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
static float rng_float(Rng *r) { return (float)((double)(rng_next(r) >> 11) / 9007199254740992.0); }

static int pop64(uint64_t x) {
  int n = 0;
  while (x) {
    x &= x - 1;
    n++;
  }
  return n;
}

static void place_food(Env *e) {
  int free_cells = CELLS - e->len;
  if (free_cells <= 0) {
    e->food = -1;
    return;
  }
  int target = rng_int(&e->rng, free_cells), seen = 0;
  for (int i = 0; i < CELLS; i++) {
    if (!e->occ[i]) {
      if (seen == target) {
        e->food = i;
        return;
      }
      seen++;
    }
  }
}

static void env_reset(Env *e) {
  memset(e->occ, 0, sizeof e->occ);
  int cx = SIZE / 2, cy = SIZE / 2;
  e->len = e->max_len = 3;
  e->dir = 1;
  e->alive = 1;
  e->score = e->steps = e->since_food = 0;
  e->visited = 0;
  for (int i = 0; i < e->len; i++) {
    e->xs[i] = cx - i;
    e->ys[i] = cy;
    int c = cell(e->xs[i], e->ys[i]);
    e->occ[c] = 1;
    e->visited |= 1ULL << c;
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
  static const int d[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
  *dx = d[dir][0];
  *dy = d[dir][1];
}

static int next_dir(int dir, int action) {
  if (action == 0) return (dir + 3) & 3;
  if (action == 2) return (dir + 1) & 3;
  return dir;
}

static int idle_limit(Env *e) {
  int v = 32 + e->len * 2;
  return v < 96 ? v : 96;
}

static int collides(Env *e, int x, int y, int grow) {
  if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return 1;
  int c = cell(x, y);
  if (!e->occ[c]) return 0;
  int tail = e->len - 1;
  return grow || x != e->xs[tail] || y != e->ys[tail];
}

static float env_step(Env *e, int action) {
  int nd = next_dir(e->dir, action), dx, dy;
  delta(nd, &dx, &dy);
  int nx = e->xs[0] + dx, ny = e->ys[0] + dy;
  int grow = e->food == cell(nx, ny);
  e->steps++;
  if (collides(e, nx, ny, grow)) {
    e->alive = 0;
    return -1.0f;
  }

  int nc = cell(nx, ny);
  e->dir = nd;
  if (grow) {
    e->len++;
    if (e->len > e->max_len) e->max_len = e->len;
  } else {
    e->occ[cell(e->xs[e->len - 1], e->ys[e->len - 1])] = 0;
  }
  for (int i = e->len - 1; i > 0; i--) {
    e->xs[i] = e->xs[i - 1];
    e->ys[i] = e->ys[i - 1];
  }
  e->xs[0] = nx;
  e->ys[0] = ny;
  e->occ[nc] = 1;
  e->visited |= 1ULL << nc;

  if (grow) {
    e->score++;
    e->since_food = 0;
    place_food(e);
    return 2.0f + 0.03f * (float)e->len;
  }
  e->since_food++;
  return -0.004f;
}

static float init_weight(Rng *r, int fan_in) {
  float scale = sqrtf(2.0f / (float)fan_in);
  return (rng_float(r) * 2.0f - 1.0f) * scale;
}

static Model *model_create(int seed) {
  Model *m = calloc(1, sizeof(Model));
  if (!m) {
    perror("calloc model");
    exit(1);
  }
  Rng r = rng_create(seed);
  for (int i = 0; i < H1 * IN; i++) m->w1[i] = init_weight(&r, 16);
  for (int i = 0; i < H2 * H1; i++) m->w2[i] = init_weight(&r, H1);
  for (int i = 0; i < ACTIONS * H2; i++) m->wp[i] = init_weight(&r, H2);
  for (int i = 0; i < H2; i++) m->wv[i] = init_weight(&r, H2);
  m->b1pow = m->b2pow = 1.0f;
  return m;
}

static void fill_obs(Env *e, unsigned char *obs) {
  memset(obs, 0, ACTIVE);
  for (int i = e->len - 1; i >= 1; i--) obs[cell(e->xs[i], e->ys[i])] = 1;
  obs[cell(e->xs[0], e->ys[0])] = 2;
  if (e->food >= 0) obs[e->food] = 3;
  obs[CELLS] = (unsigned char)e->dir;
}

static void features(const unsigned char *obs, Work *w) {
  int n = 0;
  for (int i = 0; i < CELLS; i++) {
    int c = obs[i];
    if (c) w->feat[n++] = (c - 1) * CELLS + i;
  }
  w->feat[n++] = CATS * CELLS + obs[CELLS];
  w->feat_n = n;
}

static float forward(Model *m, const unsigned char *obs, Work *w) {
  features(obs, w);
  for (int j = 0; j < H1; j++) {
    float s = m->b1[j];
    float *wj = &m->w1[j * IN];
    for (int i = 0; i < w->feat_n; i++) s += wj[w->feat[i]];
    w->h1[j] = tanhf(s);
  }
  for (int j = 0; j < H2; j++) {
    float s = m->b2[j];
    float *wj = &m->w2[j * H1];
    for (int i = 0; i < H1; i++) s += wj[i] * w->h1[i];
    w->h2[j] = tanhf(s);
  }
  for (int a = 0; a < ACTIONS; a++) {
    float s = m->bp[a];
    float *wa = &m->wp[a * H2];
    for (int j = 0; j < H2; j++) s += wa[j] * w->h2[j];
    w->logits[a] = s;
  }
  float v = m->bv;
  for (int j = 0; j < H2; j++) v += m->wv[j] * w->h2[j];
  return v;
}

static void probs(const float *logits, float *p) {
  float mx = logits[0] > logits[1] ? logits[0] : logits[1];
  if (logits[2] > mx) mx = logits[2];
  float z = 0.0f;
  for (int i = 0; i < ACTIONS; i++) {
    p[i] = expf(logits[i] - mx);
    z += p[i];
  }
  for (int i = 0; i < ACTIONS; i++) p[i] /= z;
}

static int sample_action(Rng *r, const float *logits, float *lp) {
  float p[ACTIONS];
  probs(logits, p);
  float x = rng_float(r);
  int a = x < p[0] ? 0 : x < p[0] + p[1] ? 1 : 2;
  *lp = logf(p[a] + 1e-8f);
  return a;
}

static Rollout rollout_alloc(int envs, int steps) {
  int n = envs * steps;
  Rollout r;
  r.n = n;
  r.envs = envs;
  r.obs = calloc((size_t)n * ACTIVE, 1);
  r.actions = calloc(n, 1);
  r.dones = calloc(n, 1);
  r.old_logp = calloc(n, sizeof(float));
  r.rewards = calloc(n, sizeof(float));
  r.values = calloc(n, sizeof(float));
  r.last_values = calloc(envs, sizeof(float));
  r.adv = calloc(n, sizeof(float));
  r.returns = calloc(n, sizeof(float));
  if (!r.obs || !r.actions || !r.dones || !r.old_logp || !r.rewards || !r.values || !r.last_values || !r.adv || !r.returns) {
    perror("calloc rollout");
    exit(1);
  }
  return r;
}

static void rollout_free(Rollout *r) {
  free(r->obs);
  free(r->actions);
  free(r->dones);
  free(r->old_logp);
  free(r->rewards);
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
      unsigned char *obs = &r.obs[(size_t)k * ACTIVE];
      fill_obs(&envs[e], obs);
      r.values[k] = forward(m, obs, &w);
      float lp;
      int a = sample_action(&envs[e].rng, w.logits, &lp);
      float rew = env_step(&envs[e], a);
      int done = !envs[e].alive;
      int idle = envs[e].since_food >= idle_limit(&envs[e]);
      r.actions[k] = (unsigned char)a;
      r.old_logp[k] = lp;
      r.rewards[k] = idle ? -1.0f : rew;
      r.dones[k] = (unsigned char)(done || idle);
      if (done || idle) env_reset(&envs[e]);
    }
  }
  for (int e = 0; e < env_count; e++) {
    unsigned char obs[ACTIVE];
    fill_obs(&envs[e], obs);
    r.last_values[e] = forward(m, obs, &w);
  }
  return r;
}

static void advantages(Rollout *r) {
  int steps = r->n / r->envs;
  for (int e = 0; e < r->envs; e++) {
    float gae = 0.0f;
    for (int t = steps - 1; t >= 0; t--) {
      int k = t * r->envs + e;
      float nd = r->dones[k] ? 0.0f : 1.0f;
      float nv = r->dones[k] ? 0.0f : (t == steps - 1 ? r->last_values[e] : r->values[k + r->envs]);
      float delta = r->rewards[k] + 0.99f * nv * nd - r->values[k];
      gae = delta + 0.99f * 0.95f * nd * gae;
      r->adv[k] = gae;
      r->returns[k] = gae + r->values[k];
    }
  }
  double mean = 0.0, var = 0.0;
  for (int i = 0; i < r->n; i++) mean += r->adv[i];
  mean /= (double)r->n;
  for (int i = 0; i < r->n; i++) {
    double d = (double)r->adv[i] - mean;
    var += d * d;
  }
  float inv_std = 1.0f / sqrtf((float)(var / (double)r->n) + 1e-8f);
  for (int i = 0; i < r->n; i++) r->adv[i] = (r->adv[i] - (float)mean) * inv_std;
}

static void shuffle(Rng *rng, int *xs, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = rng_int(rng, i + 1), x = xs[i];
    xs[i] = xs[j];
    xs[j] = x;
  }
}

static void zero(float *x, int n) { memset(x, 0, (size_t)n * sizeof(float)); }

static float clamp_grad(float g) {
  if (g > 1.0f) return 1.0f;
  if (g < -1.0f) return -1.0f;
  return g;
}

static void adam(float *p, float *m, float *v, float *g, int n, float step) {
  for (int i = 0; i < n; i++) {
    float gi = clamp_grad(g[i]);
    m[i] = 0.9f * m[i] + 0.1f * gi;
    v[i] = 0.999f * v[i] + 0.001f * gi * gi;
    p[i] -= step * m[i] / (sqrtf(v[i]) + 1e-8f);
  }
}

static void update(Model *m, Rollout *r, Rng *rng) {
  static float gw1[H1 * IN], gb1[H1], gw2[H2 * H1], gb2[H2], gwp[ACTIONS * H2], gbp[ACTIONS], gwv[H2];
  static float dlogits[ACTIONS], dh2[H2], dh1[H1];
  int *order = malloc((size_t)r->n * sizeof(int));
  if (!order) {
    perror("malloc order");
    exit(1);
  }
  for (int i = 0; i < r->n; i++) order[i] = i;
  Work w;

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    shuffle(rng, order, r->n);
    for (int start = 0; start < r->n; start += BATCH) {
      int stop = start + BATCH < r->n ? start + BATCH : r->n;
      float scale = 1.0f / (float)(stop - start), gbv = 0.0f;
      zero(gw1, H1 * IN);
      zero(gb1, H1);
      zero(gw2, H2 * H1);
      zero(gb2, H2);
      zero(gwp, ACTIONS * H2);
      zero(gbp, ACTIONS);
      zero(gwv, H2);

      for (int ii = start; ii < stop; ii++) {
        int k = order[ii];
        unsigned char *obs = &r->obs[(size_t)k * ACTIVE];
        float v = forward(m, obs, &w);
        float p[ACTIONS];
        probs(w.logits, p);
        int a = r->actions[k];
        float logp = logf(p[a] + 1e-8f);
        float ratio = expf(logp - r->old_logp[k]);
        float adv = r->adv[k];
        float clipped = adv >= 0.0f ? fminf(ratio, 1.2f) : fmaxf(ratio, 0.8f);
        int use_policy = ratio * adv <= clipped * adv;

        memset(dlogits, 0, sizeof dlogits);
        if (use_policy) {
          for (int i = 0; i < ACTIONS; i++) {
            float y = i == a ? 1.0f : 0.0f;
            dlogits[i] -= ratio * adv * (y - p[i]) * scale;
          }
        }
        float entropy = 0.0f;
        for (int i = 0; i < ACTIONS; i++) entropy -= p[i] * logf(p[i] + 1e-8f);
        for (int i = 0; i < ACTIONS; i++) dlogits[i] += 0.01f * p[i] * (logf(p[i] + 1e-8f) + entropy) * scale;

        memset(dh2, 0, sizeof dh2);
        for (int i = 0; i < ACTIONS; i++) {
          gbp[i] += dlogits[i];
          float *gwa = &gwp[i * H2], *wa = &m->wp[i * H2];
          for (int j = 0; j < H2; j++) {
            gwa[j] += dlogits[i] * w.h2[j];
            dh2[j] += dlogits[i] * wa[j];
          }
        }

        float dv = 0.5f * (v - r->returns[k]) * scale;
        gbv += dv;
        for (int j = 0; j < H2; j++) {
          gwv[j] += dv * w.h2[j];
          dh2[j] += dv * m->wv[j];
        }

        memset(dh1, 0, sizeof dh1);
        for (int j = 0; j < H2; j++) {
          float dz = dh2[j] * (1.0f - w.h2[j] * w.h2[j]);
          gb2[j] += dz;
          float *gwj = &gw2[j * H1], *wj = &m->w2[j * H1];
          for (int i = 0; i < H1; i++) {
            gwj[i] += dz * w.h1[i];
            dh1[i] += dz * wj[i];
          }
        }

        for (int j = 0; j < H1; j++) {
          float dz = dh1[j] * (1.0f - w.h1[j] * w.h1[j]);
          gb1[j] += dz;
          float *gwj = &gw1[j * IN];
          for (int i = 0; i < w.feat_n; i++) gwj[w.feat[i]] += dz;
        }
      }

      m->b1pow *= 0.9f;
      m->b2pow *= 0.999f;
      float step = lr * sqrtf(1.0f - m->b2pow) / (1.0f - m->b1pow);
      adam(m->w1, m->mw1, m->vw1, gw1, H1 * IN, step);
      adam(m->b1, m->mb1, m->vb1, gb1, H1, step);
      adam(m->w2, m->mw2, m->vw2, gw2, H2 * H1, step);
      adam(m->b2, m->mb2, m->vb2, gb2, H2, step);
      adam(m->wp, m->mwp, m->vwp, gwp, ACTIONS * H2, step);
      adam(m->bp, m->mbp, m->vbp, gbp, ACTIONS, step);
      adam(m->wv, m->mwv, m->vwv, gwv, H2, step);
      float g = clamp_grad(gbv);
      m->mbv = 0.9f * m->mbv + 0.1f * g;
      m->vbv = 0.999f * m->vbv + 0.001f * g * g;
      m->bv -= step * m->mbv / (sqrtf(m->vbv) + 1e-8f);
    }
  }
  free(order);
}

static int greedy(Model *m, Env *e) {
  unsigned char obs[ACTIVE];
  Work w;
  fill_obs(e, obs);
  (void)forward(m, obs, &w);
  int best = 0;
  for (int a = 1; a < ACTIONS; a++)
    if (w.logits[a] > w.logits[best]) best = a;
  return best;
}

static void eval_model(Model *m, int episodes, Eval *ev) {
  memset(ev, 0, sizeof *ev);
  for (int i = 0; i < episodes; i++) {
    Env e = env_create(50000 + i);
    while (e.alive && e.since_food < idle_limit(&e)) {
      (void)env_step(&e, greedy(m, &e));
    }
    int visits = pop64(e.visited);
    ev->mean_score += (float)e.score;
    ev->mean_peak_len += (float)e.max_len;
    ev->mean_len_cov += (float)e.max_len / (float)CELLS;
    ev->mean_visit_cov += (float)visits / (float)CELLS;
    ev->p30_len += e.max_len >= 20 ? 1.0f : 0.0f;
    ev->p30_visit += visits >= 20 ? 1.0f : 0.0f;
    ev->mean_steps += (float)e.steps;
    if (e.score > ev->best_score) ev->best_score = e.score;
    if (e.max_len > ev->best_len) ev->best_len = e.max_len;
  }
  float inv = 1.0f / (float)episodes;
  ev->mean_score *= inv;
  ev->mean_peak_len *= inv;
  ev->mean_len_cov *= inv;
  ev->mean_visit_cov *= inv;
  ev->p30_len *= inv;
  ev->p30_visit *= inv;
  ev->mean_steps *= inv;
}

static void on_sigint(int sig) {
  (void)sig;
  stop = 1;
}

static void ui_restore(void) {
  printf("\033[?25h\033[0m\n");
  fflush(stdout);
}

static void ui_start(void) {
  signal(SIGINT, on_sigint);
  atexit(ui_restore);
  printf("\033[?25l");
}

static char env_at(Env *e, int x, int y) {
  int c = cell(x, y);
  if (e->food == c) return '*';
  for (int i = 0; i < e->len; i++) {
    if (e->xs[i] == x && e->ys[i] == y) return i == 0 ? '#' : 'o';
  }
  return '.';
}

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void sleep_sec(double seconds) {
  struct timespec ts;
  ts.tv_sec = (time_t)seconds;
  ts.tv_nsec = (long)((seconds - (double)ts.tv_sec) * 1000000000.0);
  nanosleep(&ts, NULL);
}

static void show(Model *m, Env *watch, int iter, double elapsed, double samples, int envs, Eval *ev, const char *mode) {
  if (!watch->alive || watch->since_food >= idle_limit(watch)) env_reset(watch);
  (void)env_step(watch, greedy(m, watch));
  printf("\033[2J\033[Hsnake ppo c 8x8  [%s]   Ctrl+C quit\n", mode);
  printf("iter %-6d time %6.1fs   envs %-4d   %8.0f sample/s\n", iter, elapsed, envs,
         samples / (elapsed > 0.001 ? elapsed : 0.001));
  printf("score %5.2f   best %-2d   len %4.1f/64 %5.1f%%   p30 %3.0f%%\n",
         ev->mean_score, ev->best_score, ev->mean_peak_len, 100.0f * ev->mean_len_cov, 100.0f * ev->p30_len);
  printf("visit %5.1f%%   p30 %3.0f%%   steps %.0f\n\n", 100.0f * ev->mean_visit_cov, 100.0f * ev->p30_visit,
         ev->mean_steps);
  printf("+--------+\n");
  for (int y = 0; y < SIZE; y++) {
    putchar('|');
    for (int x = 0; x < SIZE; x++) putchar(env_at(watch, x, y));
    printf("|\n");
  }
  printf("+--------+\n\n# head  o body  * food");
  fflush(stdout);
}

static void train(double seconds, int env_count, int viz) {
  if (viz) ui_start();
  Model *m = model_create(8);
  Env *envs = calloc((size_t)env_count, sizeof(Env));
  if (!envs) {
    perror("calloc envs");
    exit(1);
  }
  for (int i = 0; i < env_count; i++) envs[i] = env_create(1000 + i);
  Rng rng = rng_create(99);
  Env watch = env_create(4242);
  Eval ev;
  eval_model(m, 64, &ev);

  double t0 = now_sec(), next_viz = 0.0, next_eval = 0.0, samples = 0.0;
  int iter = 0;
  while (!stop && (seconds <= 0.0 || now_sec() - t0 < seconds)) {
    Rollout r = collect(m, envs, env_count, ROLLOUT_STEPS);
    advantages(&r);
    update(m, &r, &rng);
    rollout_free(&r);
    iter++;
    samples += (double)env_count * ROLLOUT_STEPS;
    double elapsed = now_sec() - t0;
    if (elapsed >= next_eval) {
      eval_model(m, 96, &ev);
      next_eval = elapsed + 2.0;
    }
    if (viz && elapsed >= next_viz) {
      show(m, &watch, iter, elapsed, samples, env_count, &ev, "TRAIN");
      next_viz = elapsed + VIZ_DT;
    } else if (!viz) {
      printf("\riter=%d %.1fs %.0f sample/s len_cov=%.1f%% visit_cov=%.1f%%", iter, elapsed, samples / elapsed,
             100.0f * ev.mean_len_cov, 100.0f * ev.mean_visit_cov);
      fflush(stdout);
    }
  }
  eval_model(m, 512, &ev);
  printf("\nfinal seconds=%.3f iter=%d samples=%.0f sample/s=%.0f score=%.2f best=%d peak_len=%.2f best_len=%d len_cov=%.1f%% p30_len=%.1f%% visit_cov=%.1f%% p30_visit=%.1f%% steps=%.0f\n",
         now_sec() - t0, iter, samples, samples / (now_sec() - t0), ev.mean_score, ev.best_score, ev.mean_peak_len,
         ev.best_len, 100.0f * ev.mean_len_cov, 100.0f * ev.p30_len, 100.0f * ev.mean_visit_cov,
         100.0f * ev.p30_visit, ev.mean_steps);
  if (viz && seconds > 0.0 && !stop) {
    for (;;) {
      double frame = now_sec();
      show(m, &watch, iter, seconds, samples, env_count, &ev, "PREVIEW");
      double left = VIZ_DT - (now_sec() - frame);
      if (left > 0.0) sleep_sec(left);
      if (stop) break;
    }
  }
  free(envs);
  free(m);
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "train") == 0) {
    train(argc >= 3 ? atof(argv[2]) : 0.0, argc >= 4 ? atoi(argv[3]) : 256, 1);
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "bench") == 0) {
    train(argc >= 3 ? atof(argv[2]) : 30.0, argc >= 4 ? atoi(argv[3]) : 256, 0);
    return 0;
  }
  fprintf(stderr, "usage: %s {train [seconds envs]|bench [seconds envs]} (train seconds <= 0 means forever)\n", argv[0]);
  return 2;
}
