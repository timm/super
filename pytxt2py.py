#!/usr/bin/env python3
"""
pytxt2py: expand #Word markers in a .py file with the matching
'### Word' sections of a .txt file, written out as comments.
Usage: pytxt2py.py flair.py flair.txt > docs/flair.md
"""
import re, sys

def sections(txt): # "### Word\nprose..." --> {Word: prose}
  parts = re.split(r"(?m)^### +(\w+)\s*$", txt)
  return {k: v.strip() for k, v in
          zip(parts[1::2], parts[2::2])}

def expand(py, d): # "#Word" line --> commented section
  for line in py.splitlines():
    k = line.strip().lstrip("#")
    if line.strip().startswith("#") and k in d:
      yield "# ### " + k
      yield "#"
      for s in d[k].splitlines():
        yield ("# " + s).rstrip()
      yield "#"
    else:
      yield line

if __name__ == "__main__":
  py  = open(sys.argv[1]).read()
  txt = open(sys.argv[2]).read()
  print("\n".join(expand(py, sections(txt))))
