# --- demos ----------------------------------------------------
def test_list():
  "show the demos"
  for k, f in eg.items():
    print("%-12s %s" % (k, (f.__doc__ or "").strip()))

def test_all():
  "run all the demos; exit 1 if any crash"
  bad = 0
  for k, f in eg.items():
    if f not in (test_all, test_push):
      print("\n#", k); bad += run(f)
  print("\n%s failure(s)" % bad)
  sys.exit(bad > 0)

def test_push():
  "git commit -am saving; git push; git status"
  os.system("git commit -am saving; git push; git status")

def test_the():
  "show current settings"
  oo(the)

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
  s, at, v = min(cutsTbl(t, t.rows, lambda r: disty(t, r)))
  print(t.cols.names[at], "at", v, "score %.3f" % s)

def errs(t, tr, tt, rows): # prediction errors, one holdout
  return ((abs(guess(tt, tr, r) - disty(tr, r))
           for r in rows))

def test_predict():
  "20 runs: mu, sd of |predicted - actual| d2h"
  t, err = adds(csv(the.file), Tbl()), Num()
  for _ in range(20):
    rows = shuffle(t.rows)
    n = len(rows)*2//3
    tr = clone(t, rows[:n])
    adds(errs(t, tr, growTree(tr), rows[n:]), err)
  print("err mu %.3f sd %.3f" % (err.mu, err.sd))

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
      tt = growTree(tr)
      add(nleaf, tt.leafs)
      adds(errs(t, tr, tt, rows[n:]), err)
    print("%-22s %5s %4.0f err %2.0f (%2.0f)" %
          (f.split("/")[-1][:22], len(t.rows), nleaf.mu,
           100*err.mu, 100*err.sd))

def opt1(f): # one dataset: win of tree, rand, best picks
  t = adds(csv(f), Tbl())
  w = wins(t)
  ts, rs, bs = [], [], []
  for _ in range(20):
    got, test, tr = holdout(t)
    y = lambda r: disty(tr, r)
    ts += [w(got)]
    rs += [w(min(some(test, the.budget), key=y))]
    bs += [w(min(test, key=y))]
  treat, rand, best = adds(ts), adds(rs), adds(bs)
  d = 0 if same(ts, rs) else treat.mu - rand.mu
  print("%-22s %5s tree %4.0f rand %4.0f best %4.0f diff %4.0f"
        % (f.split("/")[-1][:22], len(t.rows),
           treat.mu, rand.mu, best.mu, d))

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
  t = adds(csv(the.file), Tbl())
  showTree(growTree(t), t)

def test_klass():
  "classify diabetes: accuracy of a klass tree, 5 holdouts"
  t = adds(csv("/Users/timm/gits/moot/classify/diabetes.csv"),
           Tbl())
  y = lambda r: r[t.cols.klass]
  acc = Num()
  for _ in range(5):
    rows = shuffle(t.rows)
    n = len(rows)*2//3
    tt = growTree(clone(t, rows[:n]), y=y, acc=Sym)
    add(acc, sum(predict(tt, r) == y(r)
                 for r in rows[n:])/(len(rows) - n))
  print("accuracy mu %.2f sd %.2f" % (acc.mu, acc.sd))

# --- main ------------------------------------------------------
eg = {"-" + k[5:]: f for k, f in globals().items()
      if k.startswith("test_")}

def run(f): # reseed, call f, catch crashes; 1 if crashed
  seed(the.seed)
  try: f()
  except Exception: traceback.print_exc(); return 1
  return 0

def runs():
  errs = 0
  for j, s in enumerate(sys.argv): 
    if s=="-h": print(__doc__)
    if f := eg.get("-" + s.lstrip("-")): 
      errs += run(f)
    elif hasattr(the, k := s.lstrip("-")):
      setattr(the, k, atom(sys.argv[j + 1]))
  return errs

if __name__ == "__main__": sys.exit(runs())
