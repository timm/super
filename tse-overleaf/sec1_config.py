#!/usr/bin/env python3 -B
#HashBang
# pyright: reportAttributeAccessIssue=false
# pyright: reportOperatorIssue=false, reportArgumentType=false
#StaticBugs
"""
flair: explainable multi-objective active learning
(c) 2026 Tim Menzies <timm@ieee.org> MIT license

Options:

  --p=2          minkowski coefficient
  --few=256      max rows scanned inside one tree leaf
  --stop=20      stopping rule for recursive tree generation
  --lots=2048    max rows used to build any tree
  --budget=50    total labelling budget for optimizing
  --check=30     optimization: how many top picks to label
  --est=mean     holdout: leaf estimate; near(est) or mean
  --more=4       acquisition: labels per round
  --best=.66     acquisition: pool keep fraction
  --seed=1234    random number generation
  --places=2     show these decimal places
  --file=/Users/timm/gits/moot/optimize/misc/auto93.csv
"""
#SSOT #Config #Regx #DSL
import re, os, sys, glob, traceback
from math import exp, log
from random import seed, sample
from types import SimpleNamespace as Box
import sys; sys.dont_write_bytecode = True
#Nocache

BIG = 1e30

def atom(s): # string --> number or stripped string
  for f in [int,float]:
    try: return f(s)
    except ValueError: ...
  return s.strip()

the = Box(**{k: atom(v) for k, v in
             re.findall(r"(\w+)=(\S+)", __doc__ or "")})

def csv(file): # iterate a csv file's atom rows
  with open(file) as f:
    for s in f:
      if s := s.strip():
        yield [atom(x) for x in s.split(",")]
