# =============================================================================
# Mefile for generating previews of JOSS paper
# Original author: Michael Hucka <mhucka@caltech.edu>
# License: GNU General Public License v3.0 
# Original project: https://github.com/casics/dassie
# =============================================================================

# To use this Makefile, you have to install pandoc, xetex and bibstyle. You have also to clone the
# whedon repository :
#
# git clone https://github.com/openjournals/whedon.git
#
# in the same directory.

# Change the following values to match your configuration.
# .............................................................................

input	:= paper.md
output	:= paper.pdf
bib	:= joss.bib

# Main code -- no more customization variables after this point
# .............................................................................

title	:= $(shell grep title: $(input) | sed 's/title: *//' | tr -d "'")
authors := $(shell sed -n '/authors:/,/affiliations:/p' $(input) | grep name: | sed 's/- name: *//' | paste -d, -s - | sed 's/,/, /g')
repo	:= $(shell git remote get-url origin | sed 's|git@github.com:|https://github.com/|' | sed 's/\.git//')

$(output): $(input) $(bib) Makefile
	pandoc \
	-V paper_title="$(title)" \
	-V citation_author="$(authors)" \
	-V repository="$(repo)" \
	-V archive_doi="http://dx.doi.org/00.00000/zenodo.0000000" \
	-V formatted_doi="00.00000/joss.00000" \
	-V paper_url="http://joss.theoj.org/papers/" \
	-V review_issue_url="http://joss.theoj.org/papers/" \
	-V issue="0" \
	-V volume="00" \
	-V year="2020" \
	-V submitted="00 January 0000" \
	-V published="00 January 0000" \
	-V page="00" \
	-V graphics="true" \
	-V logo_path="whedon/resources/joss/logo.png" \
	-V geometry:margin=1in \
	-o $(output) \
    	--pdf-engine=xelatex \
	--filter pandoc-citeproc $(input) \
        --csl="whedon/resources/joss/apa.csl" \
	--from markdown+autolink_bare_uris \
	--template "whedon/resources/joss/latex.template"

autorefresh:;
	((ls $(input) $(bib) | entr make $(output)) &)

