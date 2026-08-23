local _ENV = setmetatable({}, {__index=_G})
if setfenv then setfenv(1, _ENV) end -- lua 5.1
the,help = {},[[
start: config and a tiny test-runner CLI.
(c) 2026 Tim Menzies <timm@ieee.org> MIT license

Options:

  -h             show help
  --p=2          minkowski coefficient
  --stop=4       stopping rule for recursive tree generation
  --few=256      sub-sample size for pole finding
  --k=5          nearest neighbors used in a leaf
  --check=5      optimization: how many top picks to evaluate
  --bins=16      number of bins for discretization
  --cliffs=.197  cliffs delta: max effect size for "same"
  --cohen=.35    cohen d: max mean separation for "same"
  --seed=1234    random number generation
  --file=/Users/timm/gits/moot/optimize/misc/auto93.csv]]

BIG = 1e30

function atom(s) -- string --> number or trimmed string
  if type(s) ~= "string" then return s end
  return tonumber(s) or s:match"^%s*(.-)%s*$" end

for k,v in help:gmatch("(%w+)=(%S+)") do the[k]=atom(v) end

function csv(file,      f) -- iterate a csv file's atom rows
  f = assert(io.open(file))
  return function(    s,t) 
    s = f:read()
    if s == nil then f:close() else
      t={}; for x in s:gmatch"([^,]+)" do t[#t+1]=atom(x) end
      return t end end end

-- -------------------------------------------------------------
Cols,Num,Sym = {},{},{}

function Sym.new() return {} end
function Num.new() return {n=0, mu=0, m2=0} end
function Col(s)    return (s:find"^%u" and Num or Sym).new() end
function Tbl(src)  return adds(src, {rows={}, cols=nil}) end

function Cols.new(names,     i,__roles)
  i = {names=names, all={}, x={}, y={}, klass=nil}
  for at,name in ipairs(names) do
    i.all[at] = Col(name)
    Cols.roles(i, name:sub(-1), at) end
  return i end

function Cols.roles(i,z,at)
  if     z == "X" then return
  elseif z == "!" then i.klass = at
  elseif z == "+" then i.y[at] = 1
  elseif z == "-" then i.y[at] = 0
  else   i.x[at] = at end end 

function adds(src,i) -- add all from a list or iterator
  i = i or Num()
  if type(src) == "table"
  then for _,x in ipairs(src) do add(i,x) end
  else for x in src           do add(i,x) end end
  return i end

function add(i,v,  w)
  if v == "?" then return v end
  w = w or 1
  if     i.mu then welford(i,v,w)
  elseif i.all then for j,c in pairs(i.all) do add(c,v[j],w) end
  elseif i.rows then 
    if   i.cols 
    then i.rows[1+#i.rows]=v; add(i.cols, v,   w)
    else i.cols = Cols.new(v) end
  else   i[v] = w + (i[v] or 0) end
  return v end

function welford(i,v,w,      d)
  i.n  = i.n + w
  if i.n <= 1 then i.n, i.mu, i.m2 = 0,0,0 else
    d    = v - i.mu
    i.mu = i.mu + w*d/i.n
    i.m2 = i.m2 + w*d*(v - i.mu) end end

function sd(num) -- diversity of a Num
  return num.n < 2 and 0 or (num.m2/(num.n - 1))^0.5 end

function norm(num,v,      z) -- value --> 0..1, logistic cdf
  if v == "?" then return v end
  z = (v - num.mu)/(sd(num) + 1/BIG)
  return 1/(1 + math.exp(-1.7*math.max(-3, math.min(3, z)))) end

function unnorm(c,p) -- cdf mass p --> value; norm's inverse
  return c.mu + sd(c)*math.log(p/(1 - p))/1.7 end

-- -------------------------------------------------------------
function distx(t,r1,r2,      d,n,p) -- x-column distance
  d, n, p = 0, 1/BIG, the.p
  for at,_ in pairs(t.cols.x) do
    n = n + 1
    d = d + distx1(t.cols.all[at], r1[at], r2[at])^p end
  return (d/n)^(1/p) end

function distx1(col,a,b) -- helper for one column
  if a=="?" and b=="?" then return 1 end
  if col.mu then
    a,b = norm(col,a), norm(col,b)
    if a=="?" then a = b < 0.5 and 1 or 0 end
    if b=="?" then b = a < 0.5 and 1 or 0 end
    return abs(a - b) end
  return a==b and 0 or 1 end

function poles(t,      rows,far,a,b,c) -- fastmap projector
  rows = some(t.rows, the.few)
  far  = function(r) return keysort(rows,
                  function(z) return distx(t,z,r) end)[#rows] end
  a = far(rows[1])
  b = far(a)
  c = distx(t,a,b) + 1/BIG
  return function(r)
    return (distx(t,a,r)^2 + c*c - distx(t,b,r)^2)/(2*c) end
end

function halves(t,      cl,cr,nl,nr,d,v)
  -- median-split on the fastmap line; bin rows as they land
  cl,cr,nl,nr = {},{},0,0
  for at,_ in pairs(t.cols.x) do cl[at],cr[at] = {},{} end
  for j,row in ipairs(keysort(t.rows, poles(t))) do
    if j <= floor(#t.rows/2)
    then d,nl = cl, nl + 1
    else d,nr = cr, nr + 1 end
    for at,_ in pairs(t.cols.x) do
      v = bin(t.cols.all[at], row[at])
      if v ~= "?" then
        d[at][v] = 1 + (d[at][v] or 0) end end end
  return cl,cr,nl,nr end

function bin(c,v) -- top-level col c, value v --> bin id
  if v=="?" or not c.mu then return v end
  return math.min(the.bins, 1 + floor(norm(c,v)*the.bins)) end

function cutmax(      best,score) -- argmax of b^2/(b+r)
  score = function(b,r) return b*b/(b + r + 1/BIG) end
  return function(txt,b,r,      s) -- scores both directions
    if txt == nil then return best end
    s = math.max(score(b,r), score(r,b))
    if best == nil or s > best.score then
      best = {txt=txt, score=s} end end end

function syms(t,at,dl,dr,nl,nr,best,      ks) -- sym spans
  ks = {}
  for k in pairs(dl) do ks[k] = true end
  for k in pairs(dr) do ks[k] = true end
  for k in pairs(ks) do
    best(t.cols.names[at].." == "..k,
         (dl[k] or 0)/nl, (dr[k] or 0)/nr) end end

function cuts(t,at,c,dl,dr,nl,nr,best,      B,R,v) -- num spans
  B,R = 0,0
  for j = 1, the.bins - 1 do
    B = B + (dl[j] or 0)/nl
    R = R + (dr[j] or 0)/nr
    v = say(unnorm(c, j/the.bins))
    best(t.cols.names[at].." <= "..v, B, R)
    best(t.cols.names[at].." > "..v, 1-B, 1-R) end end

function contrasts(t,      cl,cr,nl,nr,best,c) -- best of all
  cl,cr,nl,nr = halves(t)
  best = cutmax()
  for at,_ in pairs(t.cols.x) do
    c = t.cols.all[at]
    if c.mu
    then cuts(t,at,c,cl[at],cr[at],nl,nr,best)
    else syms(t,at,  cl[at],cr[at],nl,nr,best) end end
  return best() end

-- -------------------------------------------------------------
abs,floor,min,max = math.abs,math.floor,math.min,math.max
cat = table.concat

SEED = the.seed
function rand(lo,hi) -- pseudo-random lo..hi (default 0..1)
  lo, hi = lo or 0, hi or 1
  SEED = (16807 * SEED) % 2147483647
  return lo + (hi - lo) * SEED / 2147483647 end

function rint(lo,hi) -- pseudo-random integer lo..hi
  return math.floor(0.5 + rand(lo,hi)) end

function sort(t,f) table.sort(t,f); return t end

function keysort(t,fn,      u) -- sorted copy of t, order fn
  u={}; for j,v in ipairs(t) do u[j] = v end
  return sort(u, function(x,y) return fn(x) < fn(y) end) end

function some(t,n,      u) -- n random picks from list t
  u={}; for j = 1, min(n, #t) do u[j] = t[rint(1, #t)] end
  return u end

function say(t,      u) -- anything --> string, tidy numbers
  if type(t)=="number" then
    return (t==floor(t) and "%d" or "%.3f"):format(t) end
  if type(t)~="table" then return tostring(t) end
  u={}
  for k,v in pairs(t) do
    u[#u+1] = (#t>0 and "" or k.."=")..say(v) end
  return "{"..cat(#t==0 and sort(u) or u,", ").."}" end

-- --------------------------------------------------------------
eg = eg or {} -- demo table: eg["-x"] = function(v) ... end

function run(f,x,      ok,err) -- reseed, call eg[f], catch
  SEED = the.seed
  ok,err = xpcall(function() return eg[f](x) end,debug.traceback)
  if not ok then print(err) end end

-- --------------------------------------------------------------
function test_the() -- show current settings
  print(say(the)) end

function test_atom() -- strings coerce to numbers or strings
  assert(atom"2" == 2 and atom"2.1" == 2.1)
  assert(atom" a " == "a")
  print("'2' ->", atom"2", "| '2.1' ->", atom"2.1",
        "| ' a ' ->", "'"..atom" a ".."'") end

function test_csv() -- csv reader finds many rows in the.file
  local n = 0
  for _ in csv(the.file) do n = n + 1 end
  assert(n > 100)
  print(n, "rows") end

function test_contrasts() -- name this table's two halves
  local t = Tbl(csv(the.file))
  local z = contrasts(t)
  assert(z and z.score > 0.5)
  print(z.txt, ("score %.2f"):format(z.score)) end

eg["-the"]        = test_the
eg["-atom"]       = test_atom
eg["-csv"]        = test_csv
eg["-contrasts"]  = test_contrasts

if arg[0] and arg[0]:find"flair" then
  for j,s in ipairs(arg) do
    if eg[s] then run(s, arg[j+1])
    else s = s:gsub("^[-]+","")
         if the[s] then the[s] = atom(arg[j+1]) end end end end

return _ENV
