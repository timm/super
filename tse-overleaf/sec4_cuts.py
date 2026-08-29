# --- cut: min expected variance splits ------------------------
def matches(col, x, v): # does x fall on the yes side of cut v?
  return x == "?" or (x == v if col.it is Sym else x <= v)

def selects(z, row): return matches(z.col, row[z.at], z.v)

def score(col1, col2): # expected diversity of col1|col2 split
  n1, n2 = col1.n, col2.n
  return (div(col1)*n1 + div(col2)*n2)/(n1 + n2 + 1/BIG)

def cutsSym(xy, tot, acc): # (score,v): yes = one symbol
  for v in dict.fromkeys(x for x, _ in xy):
    lhs = adds((y for x, y in xy if x == v), acc())
    rhs = adds((y for x, y in xy if x != v), acc())
    yield score(lhs, rhs), v

def cutsNum(xy, tot, acc): # (score,v) per bound, one sweep
  xy.sort()
  lhs = acc()
  for j, (x, y) in enumerate(xy):
    add(lhs, y); add(tot, y, -1)           # rhs = shrinking tot
    if j + 1 < len(xy) and x != xy[j+1][0]:
      yield score(lhs, tot), x

def cutsTbl(tbl, rows, y, acc=Num): # (score,at,v), all x cols
  for at in tbl.cols.x:
    xy  = [(r[at], y(r)) for r in rows if r[at] != "?"]
    tot = adds((y for _, y in xy), acc())
    f   = cutsNum if tbl.cols.all[at].it is Num else cutsSym
    for s, v in f(xy, tot, acc):
      yield s, at, v

def cutTbl(tbl, rows, y, acc=Num): # best cut, as a labeled Span
  if z := min(cutsTbl(tbl, rows, y, acc), default=None):
    _, at, v = z
    c, s = tbl.cols.all[at], tbl.cols.names[at]
    eq, ne = ("==", "!=") if c.it is Sym else ("<=", ">")
    return Box(it=cutTbl, at=at, v=v, col=c,
               txt=f"{s} {eq} {v}", anti=f"{s} {ne} {v}")

# --- tree ------------------------------------------------------
def Tree(**d):
  return Box(**dict(it=Tree, n=0, rows=[], cut=None,
                    ys=None, leafs=1) | d)

def growTree(tbl, y=None, acc=Num): # min-variance splits
  y = y or (lambda r: disty(tbl, r))
  def recurse(rows):
    node = Tree(n=len(rows),rows=rows,ys=adds(map(y,rows),acc()))
    if len(rows) > the.stop:
      if z := cutTbl(tbl, rows, y, acc):
        yes, no = [], []
        for r in rows: (yes if selects(z, r) else no).append(r)
        if yes and no:
          node.cut,node.yes,node.no = z,recurse(yes),recurse(no)
          node.leafs = node.yes.leafs + node.no.leafs
    return node
  return recurse(tbl.rows)

def leaf(tree, row): # walk row down to its leaf
  while tree.cut:
    tree = tree.yes if selects(tree.cut, row) else tree.no
  return tree

def predict(tree, row): # leaf's mode (Sym) or mean (Num)
  return mid(leaf(tree, row).ys)

def guess(tree, tbl, row): # d2h of leaf row nearest to mids
  l = leaf(tree, row)
  if not hasattr(l, "est"):
    c = mids(clone(tbl, l.rows))
    l.est = disty(tbl, min(some(l.rows, the.few),
                           key=lambda r: distx(tbl, r, c)))
  return l.est

def nodes(tree, pre=None, txt=""): # walk: (node, indented txt)
  yield tree, (pre or "") + txt
  if tree.cut:
    sub = "" if pre is None else pre + "|  "
    yield from nodes(tree.yes, sub, tree.cut.txt)
    yield from nodes(tree.no,  sub, tree.cut.anti)

def showTree(tree, tbl): # y-col mids per node; +/- bad,best leaf
  ns    = list(nodes(tree))
  leafs = [n for n, _ in ns if not n.cut]
  best  = min(leafs, key=lambda n: mid(n.ys))
  worst = max(leafs, key=lambda n: mid(n.ys))
  mark  = lambda n: ("+" if n is best else
                     "-" if n is worst else "")
  printm([["", "d2h", "n",
           *[tbl.cols.names[at] for at in tbl.cols.y], ""]] +
         [[mark(n), mid(n.ys), n.n, *ymids(tbl, n.rows), txt]
          for n, txt in ns],
         "<>>" + ">"*len(tbl.cols.y))
