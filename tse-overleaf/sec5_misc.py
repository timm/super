# --- misc -----------------------------------------------------
def shuffle(t): return sample(t, len(t)) # non-mutating

def some(t, n): # n random picks from list t, no repeats
  return sample(t, min(n, len(t)))

def o(v): # tidy: round floats; boxes --> dicts, no _keys
  if type(v) is float:
    return int(v) if v == int(v) else round(v, the.places)
  if callable(v): return v.__name__
  if hasattr(v, "__dict__"):
    return {k: o(w) for k, w in vars(v).items() if k[0] != "_"}
  return v

def oo(x): print(o(x)); return x


def printm(rows, align=""): # align columns; flags "<->" per col
  rows = [[str(o(x)) for x in r] for r in rows]
  ws = [max(map(len, c)) for c in zip(*rows)]
  aligns = align.replace("-", "^").ljust(len(ws), "<")
  for r in rows:
    print("  ".join(f"{x:{a}{w}}"
                   for x, a, w in zip(r, aligns, ws)).rstrip())

def wins(t, rows=None): # grader: row --> % of gap closed
  ys = sorted(disty(t, r) for r in rows or t.rows)
  lo, b4 = ys[0], sum(ys)/len(ys)
  return lambda r: max(-100, min(100,
    100*(1 - (disty(t, r) - lo)/(b4 - lo + 1/BIG))))

def holdout(t): # train: half, capped at lots. test: the rest
  rows = shuffle(t.rows)
  n = len(rows)//2
  tr = clone(t, descends(t, rows[:n][:the.lots]))
  tt = growTree(tr)
  est = ((lambda r: predict(tt, r)) if the.est == "mean"
         else lambda r: guess(tt, tr, r))
  top = sorted(rows[n:], key=est)
  return min(top[:the.check],
             key=lambda r: disty(tr, r)), rows[n:], tr

# --- stats: are two samples of numbers the same? ---------------
def cohen(xs, ys, d=0.35): # mean gap small, in pooled sd units
  x, y = adds(xs), adds(ys)
  sd = (((x.n-1)*x.sd**2 + (y.n-1)*y.sd**2)/(x.n+y.n-2))**0.5
  return abs(x.mu - y.mu) <= d*sd

def cliffs(xs, ys, d=0.197): # sorted xs,ys: rank imbalance.
  gt = lt = j = k = 0      # j,k = ys below, at-or-below x;
  for x in xs:             # they only ever advance
    while j < len(ys) and ys[j] <  x: j += 1; k = j
    while k < len(ys) and ys[k] <= x: k += 1
    gt += j; lt += len(ys) - k
  return abs(gt - lt)/(len(xs)*len(ys)) <= d

def ks(xs, ys, a=1.36): # sorted xs,ys: 95% kolmogorov-smirnov
  n, m, i, j, d = len(xs), len(ys), 0, 0, 0
  while i < n and j < m:
    v = min(xs[i], ys[j])
    while i < n and xs[i] <= v: i += 1
    while j < m and ys[j] <= v: j += 1
    d = max(d, abs(i/n - j/m))
  return d <= a*((n + m)/(n*m))**0.5

def same(xs, ys): # indistinguishable by all three tests
  xs, ys = sorted(xs), sorted(ys)
  return cliffs(xs, ys) and ks(xs, ys) and cohen(xs, ys)

def top(d, reverse=False): # names statistically tied with best
  mu = lambda a: sum(a)/len(a)
  xs = sorted(d.items(),key=lambda kv:mu(kv[1]), reverse=reverse)
  j = 0
  while j+1 < len(xs) and same(xs[0][1], xs[j+1][1]): j += 1
  return {k for k, _ in xs[:j+1]}
