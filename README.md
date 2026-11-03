# annotation-quality-kit

![CI](https://github.com/olusola-titilayo/annotation-quality-kit/actions/workflows/ci.yml/badge.svg)

Quality control for human-labelled AI data. It answers the three questions a data-labelling project
keeps asking: **can we trust these labels, who needs coaching, and which items need a second look?**

No dependencies. Input is a plain CSV.

```bash
pip install -e .
annotation-quality examples/labels.csv --gold examples/gold.csv
```

```
12 items, 3 annotators
Krippendorff's alpha (nominal): 0.350

annotator       items   peers   gold
ann                12    100%   100%
ben                12     86%    75%
dee                12     55%    75%

Adjudication queue: 6 item(s)
  resp-01    67% {'bad': 1, 'good': 2}
  resp-06    67% {'bad': 2, 'good': 1}
  ...
```

The example data is synthetic: three simulated annotators with different accuracy. Alpha of 0.35 is far
too low to use these labels, and the `peers` column shows where to look first.

## What it measures

| Tool | Use it when |
|---|---|
| `percent_agreement` | Never alone. It looks good on skewed data, where 90% agreement can mean no real agreement |
| `cohen_kappa` | Two annotators labelled the same items |
| `fleiss_kappa` | A fixed number of ratings per item, from any pool of annotators |
| `krippendorff_alpha` | Missing ratings, or ordered scales such as 1 to 5 (`nominal`, `ordinal`, `interval`) |
| `annotator_report` | Agreement with peers and accuracy on known-answer (gold) items, per annotator |
| `adjudication_queue` | Lists items to re-review. Items whose majority contradicts gold come first |

**Design choices worth arguing about**

- Peer agreement leaves out the annotator's own vote. Otherwise everyone gets credit for agreeing with
  themselves, which hides the weakest person on small panels.
- A unanimous answer that contradicts gold still goes in the queue. Agreement isn't correctness: everyone
  may have misread the guideline, or the gold item may be wrong.
- Peer ties aren't counted either way. There's no majority to compare with.

Use `--min-alpha 0.8` to fail a pipeline when agreement is too low. 0.8 is Krippendorff's commonly cited
threshold for reliable data, and 0.667 is the lowest for tentative conclusions.

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

The statistics are checked against **published worked examples** rather than values produced by this code:
Krippendorff (2011) nominal 0.743, ordinal 0.815, interval 0.849 and binary 0.095, the standard Fleiss
example at 0.210, and a textbook Cohen table at 0.40.

## License

MIT
