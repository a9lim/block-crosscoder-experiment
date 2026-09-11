# References

This directory keeps source lineage separate from experiment documentation and
generated research inputs.

- [`refs.yaml`](refs.yaml) is the committed bibliography and fetch manifest.
- [`literature.md`](literature.md) synthesizes the literature and scopes what
  each source does and does not support.
- [`notes/`](notes/) contains curated procedure notes for the block-method and
  same-model crosscoder families.
- `sources/` contains generated Markdown source copies and is ignored.
- `pdfs/` contains local PDF fallbacks for sources without usable HTML and is
  ignored.

Using the shared `transformer-experiments` package, refresh the generated
Markdown copies and put them in the ignored source cache with:

```bash
python -m transformer_experiments.references
find references -maxdepth 1 -type f -name '*.md' \
  ! -name README.md ! -name literature.md \
  -exec mv {} references/sources/ \;
```

Downloaded copies are reading aids, not committed evidence. Claims should cite
the primary URLs in `refs.yaml` and preserve the distinctions among paper
procedure, inspected release behavior, local adaptation, and project
hypothesis.
