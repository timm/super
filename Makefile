include .dot/Makefile

$(shell mkdir -p ~/tmp)

Cpu=10
What=contrast.py
Flag=-halve
Data=$(HOME)/gits/moot/optimize

xargs: ## run What+Flag on every Data csv, Cpu at a time
	@find $(Data) -name '*.csv' | \
	xargs -P $(Cpu) -n 1 -I{} python3 $(What) --file {} $(Flag)

~/tmp/contrast.txt: ## save the sorted xargs run here
	@$(MAKE) -s xargs | tee $@

~/tmp/%.lua.pdf: %.lua ## .lua ==> .pdf (make ~/tmp/x.lua.pdf)
	@mkdir -p ~/tmp
	@echo "pdf-ing $@ ..."
	@a2ps -BrElua --chars-per-line=80 --file-align=fill \
	  --line-numbers=1 --borders=no --pro=color --columns=3 \
	  -M letter -o - $< | ps2pdf - $@
	@open $@

## x.py + x.txt ==> docs/x.md ==> docs/x.html
.SECONDARY: # keep the intermediate .md files
docs/%.md: %.py %.txt xpand.awk fence.awk
	@mkdir -p docs
	@gawk -f xpand.awk $*.txt $*.py | gawk -f fence.awk > $@

docs/%.html: docs/%.md docs/head.html docs/body.html circle.awk
	@gawk -f circle.awk $< | \
	  pandoc -f markdown -H docs/head.html -B docs/body.html \
	    --syntax-highlighting=tango -s -o $@
	@open $@
