# Pairing Analysis APP

A production-oriented Python/Streamlit port of **GETPAIRS v2.000** (C. Neufeld, 2006) with paired-sample spatial and statistical comparison workflows.

## What it does

- Loads two datasets in GSLIB/GeoEAS (`.dat`, `.out`), CSV, or Excel format.
- Reproduces the GETPAIRS reference/comparison direction: dataset 1 drives the outer loop and is plotted on **Y**; dataset 2 is searched and plotted on **X**.
- Uses a 3D spherical maximum distance with the legacy strict predicate `distance_squared < dismax**2`.
- Supports **all neighbors within radius** or **nearest neighbor only**, preserving the two GETPAIRS `ikeepclose` modes.
- Preserves dataset-2 reuse; no one-to-one assignment is imposed.
- Reproduces the legacy shared-axis quirk and exposes a clear **2D search (ignore Z)** control.
- Displays the exact analysis-valid spatial pairs in an interactive location plot, defaulting to **XY Plan View** with optional **XZ** and **YZ** section views.
- Produces a four-panel Plotly comparison figure: KDE density scatter, CDF, Q-Q, and statistics table.
- Uses the corporate palette by default: blue `#03547C`, gold `#A39161`, orange `#FDB813`, and gray `#C7C8CA`.
- Exports paired CSV, legacy-style GSLIB, statistics CSV, Plotly HTML, PNG, and `getpairs.par`.

## Install and run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Recommended workflow

1. Upload Dataset A and Dataset B.
2. Choose which dataset is the **Reference** (outer loop / Y axis).
3. Map X, Y, Z for each dataset. Turn on **2D search (ignore Z)** when required.
4. Select the analysis variable and map the corresponding variable column in both datasets.
5. Set the maximum search distance and pairing mode.
6. Keep **valid assay only** enabled to reproduce the Pueblo Viejo study logic.
7. Optionally filter by domain/lithology or grade range.
8. Review the paired-sample location plot (XY by default; XZ/YZ available when mapped coordinates are valid).
9. Review the paired-sample comparison figure and pairing report, then download the outputs.


## Included 100-record demo datasets

The `data/` folder includes two CSV files designed specifically for testing the app with a **2.0 m** maximum search distance:

- `demo_reference_100.csv`
- `demo_comparison_100.csv`

Recommended mapping: `x`, `y`, `z`; choose `demo_reference_100.csv` as the **Reference** dataset. The files include the standard geochemical fields used by the app (`au_ppm`, `ag_ppm`, `s_tot_pct`, `s2_pct`, `c_tot_pct`, `c_org_pct`, `cu_pct`, `zn_pct`, `cao_pct`, `sio2_pct`) plus numeric `lithology` and `domain` codes. Lithology codes are `1=DIO`, `2=GAB`, `3=MDI`; domain codes are `1–3`. All fields are numeric so the paired result can also be exported to legacy GSLIB.

Expected pairing diagnostics at `dismax = 2.0 m` using the strict GETPAIRS rule (`distance < 2.0 m`):

| Mode | Pairs | Pairing rate | Unpaired reference | Reused comparison records | Mean distance | Max distance |
|---|---:|---:|---:|---:|---:|---:|
| All pairs | 93 | 90.00% | 10 | 5 | 1.07 m | 1.75 m |
| Closest only | 90 | 90.00% | 10 | 2 | 1.05 m | 1.75 m |

Five reference/comparison cases are exactly **2.000 m** apart and are deliberately excluded, and another five are **2.500 m** apart. The first ten records also contain engineered multi-candidate/reuse cases so that both pairing modes can be exercised.

## Legacy compatibility notes

### Strict radius

GETPAIRS tests:

```text
dis = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
retain if dis < dismax^2
```

A point exactly at `dismax` is excluded. The KD-tree is only a candidate-search accelerator; every candidate is rechecked using the strict squared-distance rule.

### Closest-only ties

The FORTRAN updates the closest pair only when `dis < dclose`. Therefore equal-distance ties retain the first dataset-2 record encountered. The Python implementation explicitly restores that first-occurrence rule after `cKDTree.query()`.

### Dataset-2 reuse

No 1:1 constraint exists. One comparison record may be paired to multiple reference records. The app reports how many comparison records are reused and how many extra reuse assignments occur.

### Coordinate quirk / 2D mode

In the FORTRAN, an axis is populated only when the corresponding column index is greater than zero in **both** files. Otherwise that axis is zeroed in both datasets. The engine preserves this behavior. The UI exposes the common case as **2D search (ignore Z)**.

### Assay validity / trimming

The old `tmin` filter is commented out in GETPAIRS. The app makes assay-validity and grade-range trimming explicit, user-controlled preprocessing. The default **valid assay only** setting matches the reference study statement that only valid assays for the evaluated variable were used.

## Statistics

Default study table order:

`count, mean, stdev, cv, min, P10, P50, P90, max`

`Diff.(%) = (comparison - reference) / reference * 100`

Near-zero reference denominators are suppressed and marked instead of displaying unstable percentages. The guard is scaled to the reference variable magnitude and remains available under Advanced display.

The scatter annotation coefficient `c` is implemented exactly as requested as the **Reduced Major Axis (RMA)** slope:

```text
c = sign(r) * s_reference / s_comparison
```

where `r` is the Pearson correlation.

## Tests

Run:

```bash
pytest -q
```

The core test compares the KD-tree pair set against a literal O(n1×n2) brute-force implementation, including strict-radius exclusion, closest-mode tie handling, record reuse, and the shared-axis 2D quirk.

## Project structure

The premium version uses a layered application structure: a minimal Streamlit entry point, a UI layer, an analytical core, and an independent test suite.

```text
app.py                    # Streamlit entry point only
src/
  core/
    config.py             # getpairs.par import/export
    io.py                 # GSLIB/CSV/Excel I/O and paired exports
    pairing.py            # brute-force reference + cKDTree engine
    statistics.py         # RMA/correlation/CDF/Q-Q/table metrics
    plotting.py           # spatial pair-location plot + report-grade comparison figure
  ui/
    page.py               # application workflow and controls
    theme.py              # premium corporate UI components/CSS
    palettes.py           # corporate and scientific palettes
tests/                    # algorithm and I/O validation
data/                     # synthetic/demo datasets
.streamlit/config.toml    # Streamlit theme defaults
TECHNICAL_NOTE.md
requirements.txt
```

The analytical core has no dependency on Streamlit; the UI calls the core rather than embedding pairing or statistics logic in the page code.

## GSLIB export limitation

Legacy GSLIB/GeoEAS data rows are numeric. If CSV/Excel inputs contain non-numeric fields (for example string hole IDs), the app still exports the full paired CSV but disables legacy GSLIB output until those fields are numeric or removed upstream.


## Author

Julio Solano
