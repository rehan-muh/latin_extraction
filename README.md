# Latin Poetic Word Order Extraction

Reproducible extraction and Bayesian analysis of Latin poetic word order, metrical placement, caesurae, elision, and dependency order.

## Data layers and annotation provenance

- **Pedecerto XML**: editorial text and bibliographic metadata are marked `editorial_curated`; metre, scansion, syllable positions, word-boundary codes, and elision are marked `automatic_pedecerto`.
- **Tesserae LatinPipe syntax database**: tokens, lemmas, UPOS, morphology, heads, and dependency relations are marked `automatic_latinpipe`.
- **Universal Dependencies Latin treebanks**: released treebank annotations are marked `manual_gold_ud` (while retaining each treebank's own documentation and provenance).
- All derived variables are marked `derived_from_*` and retain their source annotation fields.

The GitHub Actions workflow downloads the public/authorized source data, parses all available Pedecerto XML files, aligns Pedecerto lines with LatinPipe lines when possible, extracts relation-level syntax observations, runs Bayesian hierarchical models, and uploads a complete analysis artifact.

## Outputs

The workflow artifact contains:

- `pedecerto_lines.csv.gz`
- `pedecerto_words.csv.gz`
- `latinpipe_texts.csv.gz`
- `latinpipe_lines_poetry.csv.gz`
- `syntax_dependencies.csv.gz`
- `meter_syntax_aligned_lines.csv.gz`
- `ud_gold_tokens.csv.gz`
- corpus and annotation summaries
- posterior draws and summaries for each fitted model
- diagnostic figures
- a Markdown results report

Raw source archives are not committed to this repository.
