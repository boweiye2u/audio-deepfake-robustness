# ASVspoof 2019 Logical Access data preparation

The primary corpus is the official ASVspoof 2019 Logical Access release:

- DOI: https://doi.org/10.7488/ds/2555
- Official public record: https://zenodo.org/records/6906306
- Official download: https://zenodo.org/api/records/6906306/files/LA.zip/content
- Archive: `LA.zip`
- Published archive size: 7,640,952,520 bytes (about 7.12 GiB; displayed as 7.6 GB)
- Published MD5: `30c98f11d8b2bc21f2c257bfd78bb5c5`
- License: Open Data Commons Attribution License (ODC-By)

The release contains official train, development, and evaluation partitions with
speaker-disjoint source speech. The project preserves those partitions exactly and
never infers or regenerates a split.

## Local placement

```text
~/audio/cache/downloads/asvspoof2019/LA.zip
~/audio/data/asvspoof2019/LA/
~/audio/data/asvspoof2019/manifests/development/{train,dev}.csv
~/audio/data/asvspoof2019/manifests/final-evaluation/eval.csv
```

Dataset audio, original protocols, the archive, and full manifests stay outside Git.
Manifest paths are relative to `AUDIO_DATA_ROOT`, so moving the artifact root does not
invalidate them. Only preparation code, schema documentation, aggregate counts, and
small audit summaries belong in the code repository.

## Leakage policy

- Train only on the official training partition.
- Select models, thresholds, and corruption policies only with development data.
- Keep evaluation labels unavailable to training and selection code.
- Run the official evaluation split only after the development protocol is frozen.
- Retain speaker ID and attack ID in manifests for overlap and subgroup audits.

The evaluation manifest is physically separated from development manifests. The
manifest reader rejects evaluation rows unless the caller explicitly declares the
`final_evaluation` purpose. Aggregate counts may be inspected during preparation;
evaluation labels must not be used for model selection or exploratory tuning.

## Preparation

```bash
conda activate "$HOME/audio/envs/audio-deepfake"
python scripts/prepare_asvspoof2019_la.py plan
python scripts/prepare_asvspoof2019_la.py all
```

The downloader retains interrupted data as `LA.zip.part` and resumes it when the
official server honors byte ranges. It verifies the published MD5 before extraction,
refuses a mismatched archive, preserves the original ZIP, checks archive paths before
extraction, and builds manifests from the official protocol files.
