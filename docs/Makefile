# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean clean-cache html html-strict linkcheck check-links

clean:
	rm -rf $(BUILDDIR)/html $(BUILDDIR)/doctrees $(BUILDDIR)/linkcheck

clean-cache:
	rm -rf $(BUILDDIR)/.jupyter_cache

check-links:
	bash ../scripts/check_doc_links.sh

html: check-links
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

html-strict: check-links
	NB_EXECUTION_MODE=force $(SPHINXBUILD) -W -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
