# Publication data

`publication-data.tar.gz` is the curated numerical evidence needed to verify
the reported results and regenerate derived artifacts. `manifest.json` gives a
SHA-256 digest and byte size for all 116 members, plus a digest for the archive
itself.

The root reproduction command extracts the archive automatically. To verify or
extract it manually:

```sh
PYTHONPATH=scripts python -m validation.data_archive check
PYTHONPATH=scripts python -m validation.data_archive extract
```

The data are licensed under CC BY 4.0; cite the archived software/data release
when using them.
