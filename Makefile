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

docs/flair.md: flair.py flair.txt pytxt2py.py
	@mkdir -p docs
	@python3 pytxt2py.py flair.py flair.txt > $@
