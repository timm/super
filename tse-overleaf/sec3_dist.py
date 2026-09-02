# --- dist: distance -------------------------------------------
def _dist(col, a, b): # one column's distance
  if a == "?" and b == "?": return 1
  if col.it is Sym: return a != b
  a, b = norm(col, a), norm(col, b)
  if a == "?": a = 1 if b < 0.5 else 0
  if b == "?": b = 1 if a < 0.5 else 0
  return abs(a - b)

def distx(tbl, r1, r2): # x-column distance
  d, n = 0, 1/BIG
  for at in tbl.cols.x:
    n += 1
    d += _dist(tbl.cols.all[at], r1[at], r2[at])**the.p
  return (d/n)**(1/the.p)

def disty(tbl, row): # d2h: distance of goals to best corner
  d = sum(abs(norm(tbl.cols.all[at], row[at]) - w)**the.p
          for at, w in tbl.cols.y.items())
  return (d/len(tbl.cols.y))**(1/the.p)

def ymids(tbl, rows): # mids of the y columns, in these rows
  return [mid(adds((r[at] for r in rows if r[at] != "?"),
                   tbl.cols.all[at].it())) for at in tbl.cols.y]

# --- descend: label a few rows, cull toward the good pole ------
def poles(tbl, rows, y): # far pair in rows; best-->worst axis
  far = lambda r: max(rows, key=lambda x: distx(tbl, x, r))
  a = far(rows[0]); z = far(a)
  if y(z) < y(a): a, z = z, a
  c = distx(tbl, a, z) + 1/BIG
  return lambda r: (distx(tbl,a,r)**2 + c*c -
                    distx(tbl,z,r)**2)/(2*c)

def descends(tbl, rows, label=lambda row: row):
  def descend(rows): # one greedy descent along project
    while len(rows) > the.stop and len(lab) < cap:
      todo, more = [], min(the.more, cap - len(lab))
      for r in rows:
        if id(r) in lab: todo += [lab[id(r)]]
        elif more > 0:
          more -= 1; lab[id(r)] = label(r); todo += [lab[id(r)]]
      rows = sorted(rows, key=project(todo))
      rows = rows[:int(the.best*len(rows))]

  y       = lambda r: disty(tbl, r)
  project = lambda rows: poles(tbl, rows, y)
  lab     = {}
  cap     = the.budget - the.check
  while len(lab) < cap:
    n = len(lab)
    descend(shuffle(rows))
    if len(lab) == n: break                    # no progress
  return sorted(lab.values(), key=y)
