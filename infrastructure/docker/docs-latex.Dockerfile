# syntax=docker/dockerfile:1.7
# AgroSatCopilot - Imagen LaTeX para compilar los PDFs de docs/final_doc.
#
# Objetivo: que `make docs-pdf-docker` genere los PDFs (Avance7 ES + EN) sin que
# cada integrante del equipo instale MiKTeX/TeX Live manualmente. El motor vive
# dentro del contenedor; el repo se monta en runtime (no se hace COPY), por lo
# que la imagen no necesita reconstruirse al editar los .tex.
#
# Conjunto de paquetes texlive elegido para cubrir el preambulo de los .tex:
#   graphicx, hyperref, url, booktabs, amsmath/amssymb/amsfonts, nicefrac,
#   microtype, fancyhdr, xcolor, enumitem, multirow, tabularx, fontenc T1,
#   babel english + spanish y la fuente Times (psnfss) que carga PRIMEarxiv.sty.
#
# Uso (ver target `docs-pdf-docker` en el Makefile):
#   docker build -f infrastructure/docker/docs-latex.Dockerfile \
#     -t agrosat-docs-latex:dev infrastructure/docker
#   docker run --rm -v "<repo>:/repo" -w /repo/docs/final_doc \
#     agrosat-docs-latex:dev <comando pdflatex>
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
        texlive-lang-spanish \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo/docs/final_doc

# El loop de compilacion vive en el CMD (lo parsea Docker del lado Linux), por lo
# que `make docs-pdf-docker` no pasa script alguno en la linea de comandos del host
# y evita problemas de comillas entre cmd.exe / PowerShell / bash. Dos pasadas por
# documento para resolver \ref y \cite contra la bibliografia inline.
CMD ["sh", "-c", "for f in Avance7_equipo17 Avance7_equipo17_english; do echo \"=== Compilando $f ===\"; pdflatex -interaction=nonstopmode -halt-on-error -file-line-error $f.tex && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error $f.tex || exit 1; done; echo \"PDFs generados en $(pwd)\""]
