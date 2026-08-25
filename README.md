# Pairing Analysis APP

A production-oriented Python/Streamlit application for spatial paired-sample analysis, based on the FORTRAN **GETPAIRS** program.

The application combines legacy-compatible spatial pairing with modern statistical comparison, interactive spatial QA/QC, categorical filtering, and export-ready reporting.

## Main capabilities

- Loads two datasets in GSLIB/GeoEAS (`.dat`, `.out`), CSV, or Excel format.
- Preserves the GETPAIRS reference/comparison direction:
  - **Reference** dataset drives the outer loop and is plotted on **Y**.
  - **Comparison** dataset is searched and plotted on **X**.
- Uses a spherical search radius with the strict legacy predicate `distance_squared < dismax^2`.
- Supports **All neighbors within radius** (`ikeepclose=0`) and **Nearest neighbor only** (`ikeepclose=1`).
- Allows comparison records to be reused by multiple reference records; no global 1:1 assignment is imposed.
- Supports full 3D search or **2D search (ignore Z)**.
- Provides explicit variable mapping for both datasets.
- Supports optional valid-assay, grade-range, and **categorical variable** filtering.
- Displays the exact analysis-valid paired samples used in the calculations.
- Provides an interactive spatial pair-location figure with **XY Plan View** by default and optional **XZ** and **YZ** section views.
- Adds map-style elements to the XY view: north arrow, automatic scale bar, coordinate grid, pair connectors, legend, and equal spatial aspect ratio.
- Produces a report-grade paired-sample comparison figure containing KDE density scatter, empirical CDFs, Q-Q plot, and comparative statistics.
- Automatically adds a descriptive figure title and a footer summarizing the active analysis filters.
- Exports paired CSV, legacy-style GSLIB, statistics CSV, Plotly HTML, PNG, and `getpairs.par`.

## Installation

Recommended Python version:

```text
Python 3.11
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Recommended workflow

1. Upload **Dataset A** and **Dataset B**.
2. Select which dataset is the **Reference** dataset.
3. Map the sample/hole ID and X, Y, Z coordinates for each dataset.
4. Enable **2D search (ignore Z)** if the analysis should be performed in plan only.
5. Select the analysis variable and map the corresponding column in both datasets.
6. Define the maximum search distance.
7. Select the pairing rule: **All neighbors within radius** or **Nearest neighbor only**.
8. Configure optional eligibility filters: valid assays, grade range, and categorical variable.
9. Enter short series names for the plots and statistics table, for example `Historical RC`, `Recent RC`, `Campaign A`, or `Campaign B`.
10. Review the **Spatial Pair Locations** figure.
11. Review the paired-sample comparison figure and Pairing Report.
12. Export the required tables, plots, or legacy parameter file.

## Categorical Variable Filter

The application can restrict the analysis to one or more categories before spatial pairing.

Typical examples include:

- Lithology
- Domain
- Alteration
- Weathering
- Sample type
- Drilling campaign
- Year
- Any other categorical or low-cardinality coded field

The categorical field can be mapped independently in Dataset A and Dataset B. This is useful when equivalent fields have different column names in the two input datasets.

Only records belonging to the selected categories remain eligible for pairing.

## Spatial Pair Locations

The spatial figure shows the **exact pairs used in the statistical analysis**, after applying the distance criterion and all active eligibility filters.

### XY Plan View

The default XY view is presented as a technical map-style figure and includes:

- X/Y coordinate grid
- north arrow
- automatic metric scale bar
- reference and comparison sample symbols
- pair-connection lines
- legend
- equal X/Y spatial scale

The figure intentionally does not use a geographic basemap because input coordinates may be local mine-grid or projected coordinates and the application does not currently require a CRS/EPSG definition.

### XZ and YZ Sections

When Z is available, the user can switch to:

- **XZ Section**
- **YZ Section**

If the pairing search is set to **2D search (ignore Z)**, Z may still be displayed for spatial context but does not participate in pair acceptance.

## Paired-Sample Comparison Figure

The main comparison figure contains four analytical components:

1. **Density scatterplot**
   - Comparison values on X
   - Reference values on Y
   - 1:1 reference line
   - sample count
   - RMA slope
   - Pearson correlation
   - Spearman correlation
   - mean pair separation
2. **Empirical CDF** for the two paired distributions.
3. **Q-Q plot** for direct quantile comparison.
4. **Statistics table** with count, mean, stdev, CV, minimum, P10, P50, P90, maximum, and percentage difference.

The figure title is generated from the selected variable and series names. A footer summarizes the active analytical conditions, including search radius, pairing rule, dimensionality, and active filters.

## Pairing logic

### Strict radius

GETPAIRS evaluates:

```text
dis = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
retain if dis < dismax^2
```

A comparison sample exactly at `dismax` is excluded.

The Python implementation uses `scipy.spatial.cKDTree` to accelerate neighbor searches, while final candidate acceptance still follows the strict squared-distance rule.

### All neighbors within radius

Equivalent to:

```text
ikeepclose = 0
```

Every comparison record satisfying the search radius is retained. One reference record may therefore generate multiple paired rows.

### Nearest neighbor only

Equivalent to:

```text
ikeepclose = 1
```

At most one comparison record is retained for each reference record: the closest candidate inside the search radius.

For exact equal-distance ties, the first comparison record encountered is retained, matching the legacy FORTRAN logic.

### Comparison-record reuse

The analysis does not impose a global one-to-one matching constraint. A comparison record may therefore be paired to more than one reference record. The Pairing Report explicitly summarizes comparison-record reuse.

### Coordinate behavior

The legacy implementation only activates a coordinate axis when the corresponding coordinate column is available in both datasets.

The modern interface preserves this behavior while exposing the common workflow explicitly through **2D search (ignore Z)**.

## Eligibility filters

### Valid assay only

When enabled, non-finite values for the selected analytical variable are removed before spatial pairing.

### Grade-range filter

The user may define a minimum and maximum analytical value. Records outside that range are excluded before pairing.

### Categorical Variable Filter

The user may select a categorical field and one or more permitted category values independently for both datasets.

All active eligibility filters are summarized in the comparison-figure footer.

## Statistics

The default statistics order is:

```text
count, mean, stdev, cv, min, P10, P50, P90, max
```

Percentage difference is calculated as:

```text
Diff.(%) = (comparison - reference) / reference * 100
```

Near-zero reference denominators are suppressed rather than displaying unstable percentage values.

The scatterplot coefficient `c` is the **Reduced Major Axis (RMA)** slope:

```text
c = sign(r) * s_reference / s_comparison
```

where `r` is the Pearson correlation coefficient.

## Outputs

The application can generate:

- Paired CSV
- Paired GSLIB/GeoEAS
- Statistics CSV
- Spatial Pair Locations HTML
- Spatial Pair Locations PNG
- Comparison Figure HTML
- Presentation-ready PNG
- `getpairs.par`

### GSLIB export limitation

Legacy GSLIB/GeoEAS rows are numeric.

If CSV or Excel inputs contain non-numeric fields, such as string sample IDs, the complete paired CSV remains available, but legacy GSLIB export may be unavailable unless those fields are converted or removed upstream.

## Application architecture

The project uses a layered Streamlit architecture:

```text
app.py
src/
  core/
    config.py
    io.py
    pairing.py
    plotting.py
    statistics.py
  ui/
    page.py
    palettes.py
    theme.py
.streamlit/
  config.toml
README.md
TECHNICAL_NOTE.md
requirements.txt
```

### Core layer

`src/core/` contains parameter-file handling, data input/output, spatial pairing, statistics, and plotting. The analytical core is independent of Streamlit.

### UI layer

`src/ui/` contains the Streamlit workflow, controls, page composition, figure palettes, and application styling.

This separation keeps the pairing and statistical logic independent from presentation code.

## Streamlit deployment

For Streamlit Community Cloud use:

```text
Repository: juliocesarsolano/Pairing_Analysis
Branch: main
Main file path: app.py
Python version: 3.11
```

No external API keys are required by the application.

## Author

**Julio Solano**
