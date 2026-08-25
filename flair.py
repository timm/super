"""
flair: contrast-set learner; fastmap halves + b^2/(b+r).
(c) 2026 Tim Menzies <timm@ieee.org> MIT license

Options:

  --p=2          minkowski coefficient
  --few=256      sub-sample size for pole finding
  --stop=20      stopping rule for recursive tree generation
  --bins=16      number of bins for discretization
  --budget=40    total labelling budget for optimizing
  --check=5      optimization: how many top picks to label
  --seed=1234    random number generation
  --file=/Users/timm/gits/moot/optimize/misc/auto93.csv
"""
import re, os, sys, glob, traceback; sys.dont_write_bytecode = True
from math import exp, log
from random import seed, choice, shuffle
from types import SimpleNamespace as o

BIG = 1e30

def atom(s): # string --> number or stripped string
  try: return int(s)
  except ValueError:
    try: return float(s)
    except ValueError: return s.strip()

the = o(**{k: atom(v)
           for k, v in re.findall(r"(\w+)=(\S+)", __doc__)})

def csv(file): # iterate a csv file's atom rows
  with open(file) as f:
    for s in f:
      if s := s.strip():
        yield [atom(x) for x in s.split(",")]

# ---------------------------------------------------------------
def Num(): return o(n=0, mu=0, m2=0)
def Sym(): return {}
def Col(s): return Num() if s[0].isupper() else Sym()
def Tbl(src): return adds(src, o(rows=[], cols=None))

def clone(tbl,rows=[]): # make new  with same structure as tbl
  return Tbl( [tbl.cols.names]+rows )

def Cols(names): # names --> columns grouped into x,y
  i = o(names=names, all=[], x={}, y={}, klass=None)
  for at, s in enumerate(names):
    i.all += [Col(s)]
    if   s[-1] == "X": pass
    elif s[-1] == "!": i.klass = at
    elif s[-1] in "+-": i.y[at] = s[-1] == "+"
    else: i.x[at] = at
  return i

def adds(src, i=None): # add all from any iterable
  i = i or Num()
  for x in src: add(i, x)
  return i

def add(i, v, w=1): # add value (or row) v, weight w
# ---------------------------------------------------------------
  if v == "?": return v
  if   type(i) is dict: i[v] = w + i.get(v, 0)
  elif hasattr(i, "mu"): welford(i, v, w)
  elif hasattr(i, "rows"):
    if i.cols: i.rows += [v]; add(i.cols, v, w)
    else: i.cols = Cols(v)
  elif hasattr(i, "x"):
    for c, x in zip(i.all, v): add(c, x, w)
  return v

def welford(i, v, w): # update a Num in place
  i.n += w
  if i.n < 1: i.n = i.mu = i.m2 = 0
  else:
    d = v - i.mu
    i.mu += w*d/i.n
    i.m2 += w*d*(v - i.mu)

def mid(c): # middle: mode (Sym), mu (Num), mids (Tbl)
  if type(c) is dict: return max(c, key=c.get)
  if hasattr(c, "mu"): return c.mu
  return [mid(col) for col in c.cols.all]

def div(c): # diversity: ent (Sym) or sd (Num)
  return ent(c) if type(c) is dict else sd(c)

def sd(c): # diversity of a Num
  return 0 if c.n < 2 else (c.m2/(c.n - 1))**0.5

def ent(d): # diversity of a Sym
  n = sum(d.values())
  return -sum(v/n*log(v/n, 2) for v in d.values() if v > 0)

def norm(c, v): # value --> 0..1, logistic cdf
  if v == "?": return v
  z = (v - c.mu)/(sd(c) + 1/BIG)
  return 1/(1 + exp(-1.7*max(-3, min(3, z))))

def disty(t, row): # d2h: distance of goals to best corner
  d, n = 0, 1/BIG
  for at, w in t.cols.y.items():
    v = norm(t.cols.all[at], row[at])
    if v != "?":
      n += 1; d += abs(v - w)**the.p
  return (d/n)**(1/the.p)

# ---------------------------------------------------------------
def some(t, n): # n random picks from list t
  return [choice(t) for _ in range(min(n, len(t)))]

def distx(t, r1, r2): # x-column distance
  d, n = 0, 1/BIG
  for at in t.cols.x:
    n += 1
    d += _distx(t.cols.all[at], r1[at], r2[at])**the.p
  return (d/n)**(1/the.p)

def _distx(c, a, b): # helper for one column
  if a == "?" and b == "?": return 1
  if type(c) is dict: return a != b
  a, b = norm(c, a), norm(c, b)
  if a == "?": a = 1 if b < 0.5 else 0
  if b == "?": b = 1 if a < 0.5 else 0
  return abs(a - b)

# ---------------------------------------------------------------
def poles(t, rows): # fastmap projector along 2 far poles
  rows = some(rows, the.few)
  far  = lambda r: sorted(rows, key=lambda z: distx(t, z, r))[-1]
  a = far(rows[0])
  b = far(a)
  c = distx(t, a, b) + 1/BIG
  return lambda r:(distx(t,a,r)**2 + c*c - distx(t,b,r)**2)/(2*c)

def bin(c, v): # top-level col c, value v --> bin id
  if v == "?" or type(c) is dict: return v
  return min(the.bins, 1 + int(norm(c, v)*the.bins))

def binned(t, row, d, lo): # count row's bins into d
  for at in t.cols.x:
    v = bin(t.cols.all[at], row[at])
    if v != "?":
      d[at][v] = 1 + d[at].get(v, 0)
      if type(t.cols.all[at]) is not dict:
        lo[at][v] = min(row[at], lo[at].get(v, row[at]))

def halves(t, rows): # median-split rows; bin as we go
  cl = {at: {} for at in t.cols.x}
  cr = {at: {} for at in t.cols.x}
  lo = {at: {} for at in t.cols.x} # min seen per num bin
  nl = nr = 0
  for j, row in enumerate(sorted(rows, key=poles(t, rows))):
    if j < len(rows)//2: d, nl = cl, nl + 1
    else:                d, nr = cr, nr + 1
    binned(t, row, d, lo)
  return cl, cr, nl, nr, lo

# ---------------------------------------------------------------
def span(name, at, op, v, b, r): # scored, self-labeling
  s = v if type(v) is str else "%.3g" % v
  return o(name=name, at=at, op=op, v=v,
           txt=f"{name} {op} {s}",
           score=max(b, r)**2/(b + r + 1/BIG))

def flip(z): # the negated span: same score, opposite test
  op = {"<": ">=", ">=": "<", "==": "!=", "!=": "=="}[z.op]
  return span(z.name, z.at, op, z.v, z.score, 0)

def spans(t, rows): # yield the spans of rows' two halves
  cl, cr, nl, nr, lo = halves(t, rows)
  def syms(at, dl, dr): # one span per symbol
    for k in dl.keys() | dr.keys():
      yield span(t.cols.names[at], at, "==", k,
                 dl.get(k, 0)/nl, dr.get(k, 0)/nr)
  def cuts(at, dl, dr, lo1): # cut sweeps the bin boundaries
    B = R = 0
    for j in range(1, the.bins):
      B += dl.get(j, 0)/nl
      R += dr.get(j, 0)/nr
      if j + 1 in lo1: # cut at a value seen in the data
        yield span(t.cols.names[at], at, "<",
                   lo1[j+1], B, R)
        yield span(t.cols.names[at], at, ">=",
                   lo1[j+1], 1 - B, 1 - R)
  for at in t.cols.x:
    if type(t.cols.all[at]) is dict:
      yield from syms(at, cl[at], cr[at])
    else:
      yield from cuts(at, cl[at], cr[at], lo[at])

def contrasts(t, rows): # best span over all x columns
  return max(spans(t, rows),
             key=lambda z: z.score, default=None)

# ---------------------------------------------------------------
def selects(z, row): # does row satisfy span z?
  v = row[z.at]
  return v != "?" and (v == z.v if z.op == "==" else
                       v != z.v if z.op == "!=" else
                       v <  z.v if z.op == "<"  else
                       v >= z.v)

def tree(t, cap=BIG): # grow subtrees, at most cap leaves
  n = [1]
  def grow(rows):
    node = o(n=len(rows), rows=rows, cut=None)
    if len(rows) > the.stop and n[0] < cap:
      if z := contrasts(t, rows):
        yes = [r for r in rows if selects(z, r)]
        no  = [r for r in rows if not selects(z, r)]
        if 0 < len(yes) < len(rows):
          n[0] += 1
          node.cut = z
          node.yes = grow(yes)
          node.no  = grow(no)
    return node
  return grow(t.rows)

def leaves(node): # how many leaves in this tree?
  return leaves(node.yes) + leaves(node.no) if node.cut else 1

def leaf(node, row): # walk row down to its leaf
  while node.cut:
    node = node.yes if selects(node.cut, row) else node.no
  return node

def guess(t, node, row, k): # cached leaf d2h estimate
  l = leaf(node, row)
  if not hasattr(l, "est"):
    c = centroid(t, l.rows)
    ds = [disty(t, r) for r in
          sorted(l.rows, key=lambda r: distx(t, r, c))[:9]]
    l.est = {j: sum(ds[:j])/len(ds[:j]) for j in (1, 5, 9)}
  return l.est[k]

def centroid(t, rows): # synthetic row: mid of each x column
  row = ["?"]*len(t.cols.names)
  for at in t.cols.x:
    c = Col(t.cols.names[at])
    for r in rows: add(c, r[at])
    row[at] = mid(c)
  return row

def predict(t, node, row, k): # mean d2h of k rows near mid
  rows = leaf(node, row).rows
  c = centroid(t, rows)
  near = sorted(rows, key=lambda r: distx(t, r, c))[:k]
  return sum(disty(t, r) for r in near)/len(near)

def show(node, pre=None, txt=""): # print tree; n at left
  print(f"{node.n:5} {pre or ''}{txt}")
  if node.cut:
    sub = "" if pre is None else pre + "|  "
    show(node.yes, sub, node.cut.txt)
    show(node.no,  sub, flip(node.cut).txt)

# ---------------------------------------------------------------
def test_the():
  "show current settings"
  print(the)

def test_atom():
  "strings coerce to numbers or stripped strings"
  assert atom("2") == 2 and atom("2.1") == 2.1
  assert atom(" a ") == "a"
  print("'2' ->", atom("2"), "| ' a ' ->", atom(" a "))

def test_csv():
  "csv reader finds many rows in the.file"
  n = sum(1 for _ in csv(the.file))
  assert n > 100
  print(n, "rows")

def test_contrasts():
  "name this table's two halves"
  t = Tbl(csv(the.file))
  z = contrasts(t, t.rows)
  assert z and z.score > 0.5
  print(z.txt, "score %.2f" % z.score)

def test_predict():
  "20 runs: mu, sd of |predicted - actual| d2h"
  names, *rows = list(csv(the.file))
  err = Num()
  for _ in range(20):
    shuffle(rows)
    n = len(rows)*2//3
    tr = Tbl([names] + rows[:n])
    tt = tree(tr)
    for row in rows[n:]:
      add(err, abs(predict(tr, tt, row, 1)
                   - disty(tr, row)))
  print("err mu %.3f sd %.3f" % (err.mu, sd(err)))

def test_err():
  "20 moot sets, 20 repeats: err mu/sd for k=1,3,5,7,9"
  fs = sorted(glob.glob(
         "/Users/timm/gits/moot/optimize/**/*.csv",
         recursive=True), key=os.path.getsize)[:20]
  for f in fs:
    names, *rows = list(csv(f))
    errs = {k: Num() for k in (1, 3, 5, 7, 9)}
    nleaf = Num()
    for _ in range(20):
      shuffle(rows)
      n = len(rows)*2//3
      tr = Tbl([names] + rows[:n])
      tt = tree(tr)
      add(nleaf, leaves(tt))
      for row in rows[n:]:
        l = leaf(tt, row).rows
        c = centroid(tr, l)
        ds = [disty(tr, r) for r in
              sorted(l, key=lambda r: distx(tr, r, c))[:9]]
        want = disty(tr, row)
        for k in errs:
          got = ds[:k] or ds
          add(errs[k], abs(sum(got)/len(got) - want))
    print("%-22s %5s %4.0f " %
          (f.split("/")[-1][:22], len(rows), nleaf.mu)
          + " ".join("k%s %2.0f (%2.0f)" %
                     (k, 100*errs[k].mu, 100*sd(errs[k]))
                     for k in errs))

def opt1(f): # one dataset: wins for k=1,5,9, rand, best
    names, *rows = list(csv(f))
    t0 = Tbl([names] + rows)
    ds0 = [disty(t0, r) for r in rows]
    lo, mu = min(ds0), sum(ds0)/len(ds0)
    win = lambda x: 100*(1 - (x - lo)/(mu - lo + 1/BIG))
    out = {k: Num() for k in (1, 5, 9)}
    rand, best = Num(), Num()
    for _ in range(20):
      shuffle(rows)
      n = len(rows)//2
      tr = Tbl([names] + rows[:n])
      tt = tree(tr, cap=the.budget - the.check)
      ds = [disty(tr, r) for r in rows[n:]]
      add(best, min(ds))
      add(rand, min(some(ds, the.check)))
      for k in out:
        picks = sorted(rows[n:], key=lambda r:
                       guess(tr, tt, r, k))[:the.check]
        add(out[k], min(disty(tr, r) for r in picks))
    print("%-22s %5s " % (f.split("/")[-1][:22], len(rows))
          + " ".join("k%s %4.0f" % (k, win(out[k].mu))
                     for k in out)
          + " rand %4.0f best %4.0f" %
            (win(rand.mu), win(best.mu)))

def test_opt():
  "optimize the 20 smallest moot data sets"
  fs = sorted(glob.glob(
         "/Users/timm/gits/moot/optimize/**/*.csv",
         recursive=True), key=os.path.getsize)[:20]
  stop0, the.stop = the.stop, 10
  for f in fs: opt1(f)
  the.stop = stop0

def test_opt1():
  "optimize just the.file (for parallel sweeps)"
  stop0, the.stop = the.stop, 10
  opt1(the.file)
  the.stop = stop0

def test_tree():
  "recursive contrast splits; leaves show row counts"
  show(tree(Tbl(csv(the.file))))

eg = {"-the": test_the, "-atom": test_atom,
      "-csv": test_csv, "-contrasts": test_contrasts,
      "-tree": test_tree, "-predict": test_predict,
      "-err": test_err, "-opt": test_opt,
      "-opt1": test_opt1}

def run(f): # reseed, call f, catch crashes
  seed(the.seed)
  try: f()
  except Exception: traceback.print_exc()

if __name__ == "__main__":
  for j, s in enumerate(sys.argv):
    if s in eg: run(eg[s])
    elif hasattr(the, s.lstrip("-")):
      setattr(the, s.lstrip("-"), atom(sys.argv[j + 1]))
