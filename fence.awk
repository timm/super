# fence.awk: comment paras --> prose; code --> ```py fences.
# RS="" reads paragraph at a time; FS="\n" lines as fields.
BEGIN       { RS = ""; FS = "\n"; ORS = "\n\n" }
/^#!/       { next }                     # drop the hashbang
$1 ~ /^# ?/ { gsub(/(^|\n)# ?/, "\n")
              sub(/^\n/, ""); print; next }
            { print "```py\n" $0 "\n```" }
