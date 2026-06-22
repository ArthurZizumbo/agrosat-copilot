# syntax=docker/dockerfile:1.7
# AgroSatCopilot - Imagen LaTeX para compilar el manuscrito del Paper Track
# (paper/main.tex, EPIC 11 / US-071).
#
# A diferencia de docs-latex.Dockerfile (Avance7 con bibliografia inline),
# el paper usa BibTeX (paper/bib/refs.bib), por lo que la secuencia es
# pdflatex -> bibtex -> pdflatex -> pdflatex. El repo se monta en runtime
# (no se hace COPY): editar los .tex no requiere reconstruir la imagen.
#
# Uso (ver target `paper-pdf-docker` en el Makefile):
#   docker build -f infrastructure/docker/paper-latex.Dockerfile \
#     -t agrosat-paper-latex:dev infrastructure/docker
#   docker run --rm -v "<repo>:/repo" -w /repo/paper agrosat-paper-latex:dev
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-plain-generic \
        texlive-lang-english \
        texlive-bibtex-extra \
        biber \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo/paper

# Secuencia con BibTeX para resolver \cite contra bib/refs.bib.
CMD ["sh", "-c", "pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex && bibtex main && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex && echo \"PDF generado en $(pwd): main.pdf\""]
