# xpand.awk: expand #Word markers in a .py file using the
# "### Word" sections of a .txt file, written as comments.
# usage: gawk -f xpand.awk flair.txt flair.py > docs/flair.md
FNR==NR { if (/^### /) k = $2
          a[k] = a[k] "# " $0 "\n"
          next }
/^# -+$/  { print ""; print; print ""; next }
/^#[A-Z]/ { print ""            # blank-pad the expansions
  for (i=1; i<=NF; i++) {
    w = substr($i, 2)
    if (w in a) print a[w] "\n"
    else { print "xpand.awk: ?? " w > "/dev/stderr"
           printf "# ### %s (no text yet)\n\n",w } }
    next }
1
