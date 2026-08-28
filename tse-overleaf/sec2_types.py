# --- create: all types are a Box, tagged by their maker -------
def Num():  return Box(it=Num, n=0, mu=0, m2=0, sd=0)
def Sym():  return Box(it=Sym, n=0, has={})
def Col(s): return (Num if s[0].isupper() else Sym)()
def Tbl():  return Box(it=Tbl, rows=[], cols=None)

def Cols(names): # names --> columns grouped into x,y
  i = Box(it=Cols, names=names, all=[], x={}, y={}, klass=None)
  for at, s in enumerate(names):
    i.all += [Col(s)]
    if   s[-1] == "X": pass
    elif s[-1] == "!": i.klass = at
    elif s[-1] in "+-": i.y[at] = s[-1] == "+"
    else: i.x[at] = at
  return i

def clone(tbl, rows=[]): # new table, same structure as tbl
  return adds([tbl.cols.names] + rows, Tbl())

# --- add: update -----------------------------------------------
def adds(src, it=None): # add all from any iterable
  it = it or Num()
  for x in src: add(it, x)
  return it

def add(i, v, w=1): # update any box; w=-1 is deletion
  if i.it is Tbl:
    if i.cols: add(i.cols, v, w); i.rows += [v]
    else: i.cols = Cols(v)
  elif i.it is Cols:
    [add(c, x, w) for c, x in zip(i.all, v) if x != "?"]
  else:
    i.n += w
    if   i.it is Sym: i.has[v] = w + i.has.get(v, 0)
    elif i.n < 1: i.n = i.mu = i.m2 = i.sd = 0
    else:
      d = v - i.mu
      i.mu += w*d/i.n
      i.m2 = max(0, i.m2 + w*d*(v - i.mu))
      i.sd = 0 if i.n < 2 else (i.m2/(i.n - 1))**0.5

# --- mid, div: central tendency, diversity ---------------------
def mid(col): # Num: mean. Sym: mode
  return (col.mu if col.it is Num else
          max(col.has, key=col.has.get))

def mids(tbl): return [mid(col) for col in tbl.cols.all]

def div(col): # Num: sd. Sym: entropy (v>0 dodges deletions)
  return col.sd if col.it is Num else \
    -sum(v/col.n*log(v/col.n,2) for v in col.has.values() if v>0)

def norm(num, v): # Num value --> 0..1, logistic cdf
  if v == "?": return v
  z = (v - num.mu)/(num.sd + 1/BIG)
  return 1/(1 + exp(-1.7*max(-3, min(3, z))))
