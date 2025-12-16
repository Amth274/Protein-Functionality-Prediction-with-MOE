# IEEE Paper: Meta-Learning Enhanced Protein Language Models for Fitness Prediction

## Files

- `main.tex` - Main LaTeX source file (IEEE conference format)
- `references.bib` - BibTeX bibliography
- `Makefile` - Build automation

## Compilation

### Using Make
```bash
make        # Compile PDF
make clean  # Remove auxiliary files
```

### Manual Compilation
```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- IEEEtran document class
- Required packages: cite, amsmath, graphicx, booktabs, hyperref

## Paper Structure

1. **Abstract** - Summary of approach and results
2. **Introduction** - Problem motivation and contributions
3. **Related Work** - PLMs, fitness prediction, meta-learning
4. **Methods** - Architecture, meta-learning training, evaluation
5. **Experiments** - Main results, ablations, protein analysis
6. **Discussion** - Key findings, limitations, future work
7. **Conclusion** - Summary of contributions

## Key Results

- **Test Spearman**: 0.6286 (SOTA: 0.62)
- **Improvement**: +47% over baseline, +1.4% over SOTA
- **Approach**: ESM2-650M + Meta-learning (no MSA, no structure)
