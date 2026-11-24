# Build the Poros paper.
#   make          values, tables and Figure 1 from the benchmark logs, then the PDF.  A value missing
#                 from the logs prints as an orange marker, and the page header says so.
#   make final    the same, but refuses (exit 1) while any value is missing, a cited log is outside the
#                 benchmark-logs repository, or a log contradicts the paper's comparison.
#   make pending  list the missing values and the log each is read from.
#   make arxiv    refresh arxiv-submission/ (fonts bundled) and test-build it.
#   make clean    remove LaTeX intermediates.
# Needs xelatex, latexmk, matplotlib, the Geist fonts (system or bundled) and the XCharter fonts in
# fonts/.  POROS_BENCHMARKS is the root of a checkout of github.com/Caistro-Labs/poros-benchmarks
# (unset: this repository's root); POROS_MAC_LOGS and POROS_LAPTOP_LOGS optionally name further
# folders of logs.
SRC := src
ARX := arxiv-submission
export POROS_BENCHMARKS POROS_MAC_LOGS POROS_LAPTOP_LOGS POROS_4090_LOGS
LATEX := latexmk -xelatex -interaction=nonstopmode -halt-on-error

.PHONY: all tables pdf final pending arxiv clean

all: pdf

tables:
	cd $(SRC) && python3 launch_results.py && python3 logs_appendix.py

pdf: tables
	cd $(SRC) && $(LATEX) main.tex
	@grep -c '^POROS-PENDING: ' $(SRC)/main.log | xargs -I{} echo "pending markers printed: {}"
	cp $(SRC)/main.pdf poros.pdf

final:
	cd $(SRC) && python3 launch_results.py --final && python3 logs_appendix.py --final
	cd $(SRC) && $(LATEX) -g -usepretex='\def\porosfinal{}' main.tex
	cp $(SRC)/main.pdf poros.pdf

pending: tables
	@cat $(SRC)/pending_values.txt

arxiv: pdf
	rm -rf $(ARX) && mkdir -p $(ARX)
	cp $(SRC)/*.tex $(SRC)/figure1_h2h.pdf $(SRC)/caistro_mark.pdf fonts/*.otf fonts/*.txt $(ARX)/
	sed -i 's|^\\providecommand{\\fontdir}{../fonts/}|\\providecommand{\\fontdir}{./}|' $(ARX)/main.tex
	grep -q 'providecommand{\\fontdir}{./}' $(ARX)/main.tex
	rm -rf .arxiv-check && cp -r $(ARX) .arxiv-check
	cd .arxiv-check && $(LATEX) main.tex >/dev/null
	rm -rf .arxiv-check

clean:
	cd $(SRC) && latexmk -C main.tex
