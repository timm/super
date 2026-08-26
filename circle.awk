# circle.awk: ^(3) --> a big filled circle numeral
{ print gensub(/\^\(([0-9]+)\)/,
        "<span class=\"num\">\\1</span>", "g") }
