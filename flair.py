"""
flair: contrast-set learner; fastmap halves + b^2/(b+r).
(c) 2026 Tim Menzies <timm@ieee.org> MIT license

Options:

  --p=2          minkowski coefficient
  --few=256      sub-sample size for pole finding
  --bins=16      number of bins for discretization
  --seed=1234    random number generation
  --file=/Users/timm/gits/moot/optimize/misc/auto93.csv
"""
import re, sys, traceback; sys.dont_write_bytecode = True
from math import exp
from random import seed, choice
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

def sd(c): # diversity of a Num
  return 0 if c.n < 2 else (c.m2/(c.n - 1))**0.5

def norm(c, v): # value --> 0..1, logistic cdf
  if v == "?": return v
  z = (v - c.mu)/(sd(c) + 1/BIG)
  return 1/(1 + exp(-1.7*max(-3, min(3, z))))

# ---------------------------------------------------------------
def keysort(t, fn): return sorted(t, key=fn)

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
def poles(t): # fastmap projector along 2 far poles
  rows = some(t.rows, the.few)
  far  = lambda r: keysort(rows, lambda z: distx(t, z, r))[-1]
  a = far(rows[0])
  b = far(a)
  c = distx(t, a, b) + 1/BIG
  return lambda r:(distx(t,a,r)**2 + c*c - distx(t,b,r)**2)/(2*c)

def bin(c, v): # top-level col c, value v --> bin id
  if v == "?" or type(c) is dict: return v
  return min(the.bins, 1 + int(norm(c, v)*the.bins))

def halves(t): # median-split on fastmap line; bin as we go
  cl = {at: {} for at in t.cols.x}
  cr = {at: {} for at in t.cols.x}
  lo = {at: {} for at in t.cols.x} # min seen per num bin
  nl = nr = 0
  for j, row in enumerate(keysort(t.rows, poles(t))):
    if j < len(t.rows)//2: d, nl = cl, nl + 1
    else:                  d, nr = cr, nr + 1
    for at in t.cols.x:
      v = bin(t.cols.all[at], row[at])
      if v != "?":
        d[at][v] = 1 + d[at].get(v, 0)
        if type(t.cols.all[at]) is not dict:
          lo[at][v] = min(row[at], lo[at].get(v, row[at]))
  return cl, cr, nl, nr, lo

def span(t, at, op, v, b, r): # scored, self-labeling span
  s = v if type(v) is str else "%.3g" % v
  return o(at=at, op=op, v=v,
           txt=f"{t.cols.names[at]} {op} {s}",
           score=max(b, r)**2/(b + r + 1/BIG))

def spans(t): # yield every candidate span of the two halves
  cl, cr, nl, nr, lo = halves(t)
  def syms(at, dl, dr): # one span per symbol
    for k in dl.keys() | dr.keys():
      yield span(t, at, "==", k,
                 dl.get(k, 0)/nl, dr.get(k, 0)/nr)
  def cuts(at, dl, dr, lo1): # cut sweeps the bin boundaries
    B = R = 0
    for j in range(1, the.bins):
      B += dl.get(j, 0)/nl
      R += dr.get(j, 0)/nr
      if j + 1 in lo1: # cut at a value seen in the data
        yield span(t, at, "<",  lo1[j+1], B, R)
        yield span(t, at, ">=", lo1[j+1], 1 - B, 1 - R)
  for at in t.cols.x:
    if type(t.cols.all[at]) is dict:
      yield from syms(at, cl[at], cr[at])
    else:
      yield from cuts(at, cl[at], cr[at], lo[at])

def contrasts(t): # best span over all x columns
  return max(spans(t), key=lambda z: z.score, default=None)

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
  z = contrasts(Tbl(csv(the.file)))
  assert z and z.score > 0.5
  print(z.txt, "score %.2f" % z.score)

eg = {"-the": test_the, "-atom": test_atom,
      "-csv": test_csv, "-contrasts": test_contrasts}

def run(f): # reseed, call f, catch crashes
  seed(the.seed)
  try: f()
  except Exception: traceback.print_exc()

if __name__ == "__main__":
  for j, s in enumerate(sys.argv):
    if s in eg: run(eg[s])
    elif hasattr(the, s.lstrip("-")):
      setattr(the, s.lstrip("-"), atom(sys.argv[j + 1]))
