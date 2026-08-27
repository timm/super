#!/usr/bin/env python3 -B
#HashBang
"""
flair: contrast-set learner; fastmap halves + b^2/(b+r).
(c) 2026 Tim Menzies <timm@ieee.org> MIT license

Options:

  --p=2          minkowski coefficient
  --few=256      sub-sample size for pole finding
  --stop=20      stopping rule for recursive tree generation
  --lots=2048    max rows used to build any tree
  --budget=40    total labelling budget for optimizing
  --check=5      optimization: how many top picks to label
  --seed=1234    random number generation
  --places=3     show these decimal places
  --file=/Users/timm/gits/moot/optimize/misc/auto93.csv
"""
#SSOT #Config #Regx #DSL
import re, os, sys, glob, traceback
from copy import copy
from math import exp, log
from random import seed, choice, sample
from types import SimpleNamespace as SNs
import sys; sys.dont_write_bytecode = True
#Nocache
def shuffle(t): return sample(t, len(t)) # non-mutating

BIG = 1e30

def atom(s): # string --> number or stripped string
  try: return int(s)
  except ValueError:
    try: return float(s)
    except ValueError: return s.strip()

the = SNs({k: atom(v)
           for k, v in re.findall(r"(\w+)=(\S+)", __doc__)})

def csv(file): # iterate a csv file's atom rows
  with open(file) as f:
    for s in f:
      if s := s.strip():
        yield [atom(x) for x in s.split(",")]

def some(t, n): # n random picks from list t
  return [choice(t) for _ in range(min(n, len(t)))]

# --- struct: instances copy class defaults; xKlass --> Klass.x
class o(SNs):
  def __init__(i, **k):
    super().__init__(**{a: copy(v) for a, v in
                        vars(type(i)).items()
                        if a[0] != '_' and not callable(v)} | k)
  def __repr__(i):
    f = lambda v: round(v, the.places) if type(v) is float else v
    return str({k:f(v) for k, v in vars(i).items() if k[0]!="_"})

def meths(g): # def addNum(i,..) --> Num.add
  for k, f in list(g.items()):
    if x := re.fullmatch(r'([a-z]+)([A-Z]\w*)', k):
      setattr(g[x[2]], x[1], f)

# --- create -----------------------------------------------------
class Num(o): n=0; mu=0; m2=0
class Sym(dict): pass
class Tbl(o): rows=[]; cols=None; center=None

def Col(s): return (Num if s[0].isupper() else Sym)()

class Cols(o): # names --> columns grouped into x,y
  def __init__(i, names):
    super().__init__(names=names, all=[], x={}, y={}, klass=None)
    for at, s in enumerate(names):
      i.all += [Col(s)]
      if   s[-1] == "X": pass
      elif s[-1] == "!": i.klass = at
      elif s[-1] in "+-": i.y[at] = s[-1] == "+"
      else: i.x[at] = at

def clone(tbl, rows=[]): # new table, same structure as tbl
  return adds([tbl.cols.names] + rows, Tbl())

# --- add: update ------------------------------------------------
def adds(src, i=None): # add all from any iterable
  i = i or Num()
  for x in src: i.add(x)
  return i

def addSym(i, v, w=1): i[v] = w + i.get(v, 0)

def addCols(i, v, w=1):
  [c.add(x, w) for c, x in zip(i.all, v) if x != "?"]

def addNum(i, v, w=1):
  i.n += w
  if i.n < 1: i.n = i.mu = i.m2 = 0
  else:
    d = v - i.mu
    i.mu += w*d/i.n
    i.m2 += w*d*(v - i.mu)

def subNum(i, j): # Num for "everything in i but not j"
  n = i.n - j.n
  if n <= 0: return Num()
  d = j.mu - i.mu
  return Num(n=n, mu=(i.n*i.mu - j.n*j.mu)/n,
             m2=max(0, i.m2 - j.m2 - d*d*i.n*j.n/n))

def addTbl(i, v, w=1):
  i.center = None
  if i.cols: 
    i.cols.add(v, w)
    i.rows += [v]
  else: i.cols = Cols(v)

# --- mid, div: central tendency, diversity ----------------------
def midSym(i): return max(i, key=i.get)
def midNum(i): return i.mu

def midTbl(i): # JIT center
  i.center = i.center or [c.mid() for c in i.cols.all]
  return i.center

def divNum(i): # standard deviation
  return 0 if i.n < 2 else (i.m2/(i.n - 1))**0.5

def divSym(i): # entropy
  n = sum(i.values())
  return -sum(v/n*log(v/n, 2) for v in i.values() if v > 0)

def normNum(i, v): # value --> 0..1, logistic cdf, memoized
  if v == "?": return v
  z = (v - i.mu)/(i.div() + 1/BIG)
  return 1/(1 + exp(-1.7*max(-3, min(3, z))))

# --- dist: distance ---------------------------------------------
def distSym(i, a, b):
  return 1 if a == "?" and b == "?" else a != b

def distNum(i, a, b):
  if a == "?" and b == "?": return 1
  a, b = i.norm(a), i.norm(b)
  if a == "?": a = 1 if b < 0.5 else 0
  if b == "?": b = 1 if a < 0.5 else 0
  return abs(a - b)

def distxTbl(i, r1, r2): # x-column distance
  d, n = 0, 1/BIG
  for at in i.cols.x:
    n += 1
    d += i.cols.all[at].dist(r1[at], r2[at])**the.p
  return (d/n)**(1/the.p)

def distyTbl(i, row): # d2h: distance of goals to best corner
  d, n = 0, 1/BIG
  for at, w in i.cols.y.items():
    v = i.cols.all[at].norm(row[at])
    if v != "?":
      n += 1; d += abs(v - w)**the.p
  return (d/n)**(1/the.p)

# --- cut: min expected variance splits --------------------------
class Span(o): pass # one cut: at, v, col, txt, anti

def selectsSpan(i, row): # does row go down the yes side?
  return i.col.has(row[i.at], i.v)

def hasSym(i, x, v): return x == "?" or x == v
def hasNum(i, x, v): return x == "?" or x <= v

def score(a, b): # expected diversity after a split
  return (a.div()*a.n + b.div()*b.n)/(a.n + b.n + 1/BIG)

def cutsSym(i, xy, tot): # (score,v) per symbol
  d = {}
  for x, y in xy: d.setdefault(x, Num()).add(y)
  if len(d) > 1:
    for v, lhs in d.items(): yield score(lhs, tot - lhs), v

def cutsNum(i, xy, tot): # (score,v) per boundary, one sweep
  xy.sort()
  lhs = Num()
  for j, (x, y) in enumerate(xy):
    lhs.add(y)
    if j + 1 < len(xy) and x != xy[j+1][0]:
      yield score(lhs, tot - lhs), x

def cutsTbl(i, rows, Y): # candidate (score,at,v), all x cols
  for at in i.cols.x:
    xy = [(r[at], Y(r)) for r in rows if r[at] != "?"]
    for s, v in i.cols.all[at].cuts(xy, adds(y for _, y in xy)):
      yield s, at, v

def cutTbl(i, rows, Y): # best cut, as a self-labeling Span
  if z := min(i.cuts(rows, Y), default=None):
    _, at, v = z
    c, s = i.cols.all[at], i.cols.names[at]
    eq, ne = ("==", "!=") if isinstance(c, Sym) else ("<=", ">")
    return Span(at=at, v=v, col=c,
                txt=f"{s} {eq} {v}", anti=f"{s} {ne} {v}")

# --- tree -------------------------------------------------------
class Tree(o): n=0; rows=[]; cut=None

def treeTbl(i, cap=BIG): # grow subtrees, at most cap leaves
  Y, n = (lambda r: i.disty(r)), [1]
  def grow(rows):
    node = Tree(n=len(rows), rows=rows)
    if len(rows) > the.stop and n[0] < cap:
      if z := i.cut(rows, Y):
        yes = [r for r in rows if z.selects(r)]
        no  = [r for r in rows if not z.selects(r)]
        if 0 < len(yes) < len(rows):
          n[0] += 1
          node.cut = z
          node.yes, node.no = grow(yes), grow(no)
    return node
  return grow(i.rows)

def leavesTree(i): # how many leaves below here?
  return i.yes.leaves() + i.no.leaves() if i.cut else 1

def leafTree(i, row): # walk row down to its leaf
  while i.cut:
    i = i.yes if i.cut.selects(row) else i.no
  return i

def guessTree(i, t, row): # d2h of leaf row nearest to mids
  l = i.leaf(row)
  if not hasattr(l, "est"):
    c = clone(t, l.rows).mid()
    l.est = t.disty(min(some(l.rows, the.few),
                        key=lambda r: t.distx(r, c)))
  return l.est

def showTree(i, pre=None, txt=""): # print tree; n at left
  print(f"{i.n:5} {pre or ''}{txt}")
  if i.cut:
    sub = "" if pre is None else pre + "|  "
    i.yes.show(sub, i.cut.txt)
    i.no.show(sub, i.cut.anti)

meths(globals())
Num.__sub__ = Num.sub

# --- demos ------------------------------------------------------
def test_list():
  "show the demos"
  for k, f in eg.items():
    print("%-12s %s" % (k, (f.__doc__ or "").strip()))

def test_all():
  "run all the demos"
  for k, f in eg.items():
    if f not in (test_all, test_push):
      print("\n#", k); run(f)

def test_push():
  "git commit -am saving; git push; git status"
  os.system("git commit -am saving; git push; git status")

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

def test_cuts():
  "best (least variance) cut of full table"
  t = adds(csv(the.file), Tbl())
  s, at, v = min(t.cuts(t.rows, lambda r: t.disty(r)))
  print(t.cols.names[at], "at", v, "score %.3f" % s)

def errs(t, tr, tt, rows): # prediction errors, one holdout
  return ((abs(tt.guess(tr, r) - tr.disty(r))
           for r in rows))

def test_predict():
  "20 runs: mu, sd of |predicted - actual| d2h"
  t, err = adds(csv(the.file), Tbl()), Num()
  for _ in range(20):
    rows = shuffle(t.rows)
    n = len(rows)*2//3
    tr = clone(t, rows[:n])
    adds(errs(t, tr, tr.tree(), rows[n:]), err)
  print("err mu %.3f sd %.3f" % (err.mu, err.div()))

def test_err():
  "20 moot sets, 20 repeats: err mu/sd, mean leaf count"
  fs = sorted(glob.glob(
         "/Users/timm/gits/moot/optimize/**/*.csv",
         recursive=True), key=os.path.getsize)[:20]
  for f in fs:
    t = adds(csv(f), Tbl())
    err, nleaf = Num(), Num()
    for _ in range(20):
      rows = shuffle(t.rows)
      n = len(rows)*2//3
      tr = clone(t, rows[:n])
      tt = tr.tree()
      nleaf.add(tt.leaves())
      adds(errs(t, tr, tt, rows[n:]), err)
    print("%-22s %5s %4.0f err %2.0f (%2.0f)" %
          (f.split("/")[-1][:22], len(t.rows), nleaf.mu,
           100*err.mu, 100*err.div()))

def wins(t, rows=None): # grader: row --> % of gap closed
  ys = sorted(t.disty(r) for r in rows or t.rows)
  lo, b4 = ys[0], sum(ys)/len(ys)
  return lambda r: max(-100, min(100,
    100*(1 - (t.disty(r) - lo)/(b4 - lo + 1/BIG))))

def holdout(t): # model built on half ranks the other half
  rows = shuffle(t.rows)
  half = len(rows)//2
  tr = clone(t, rows[:half][:the.lots])
  tt = tr.tree(cap=the.budget - the.check) # 1 lab/leaf
  top = sorted(rows[half:], key=lambda r: tt.guess(tr, r))
  return min(top[:the.check],
             key=lambda r: tr.disty(r)), rows[half:], tr

def opt1(f): # one dataset: win of tree, rand, best picks
  t = adds(csv(f), Tbl())
  W = wins(t)
  treat, rand, best = Num(), Num(), Num()
  for _ in range(20):
    got, test, tr = holdout(t)
    Y = lambda r: tr.disty(r)
    treat.add(W(got))
    rand.add(W(min(some(test, the.budget), key=Y)))
    best.add(W(min(test, key=Y)))
  print("%-22s %5s tree %4.0f rand %4.0f best %4.0f" %
        (f.split("/")[-1][:22], len(t.rows),
         treat.mu, rand.mu, best.mu))

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
  adds(csv(the.file), Tbl()).tree().show()

eg = {"-" + k[5:]: f for k, f in globals().items()
      if k.startswith("test_")}

def run(f): # reseed, call f, catch crashes
  seed(the.seed)
  try: f()
  except Exception: traceback.print_exc()

if __name__ == "__main__":
  for j, s in enumerate(sys.argv):
    if f := eg.get("-" + s.lstrip("-")): run(f)
    elif hasattr(the, k := s.lstrip("-")):
      setattr(the, k, atom(sys.argv[j + 1]))
