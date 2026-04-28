# Data Cache

`data_cache.json` is a local, untracked cache of the DCM API response. It may
contain raw expert identifiers and should not be committed.

The ordinal runner reads the cache if it exists:

```bash
python run_gwt_ordinal_baseline.py --config hard_anchor_3s_paper_tree
```

If the cache is missing, the runner fetches the configured DCM API endpoint and
writes a new local `data_cache.json`. Tests do not use the live API; they use
synthetic fixtures only.

Each production run writes metadata beside the generated inference data. The
metadata records:

- git SHA
- explicit config name
- sampling and PPC seeds
- sampling budget and `target_accept`
- package versions
- data source
- canonical SHA-256 hash of the data snapshot
- timestamp
- anonymized expert labels only

Keep any private mapping between raw expert identifiers and anonymized labels
outside the repository.
