# Technical Note — GETPAIRS v2.000 Python Port and Algorithm Validation

## 1. Conclusion

The proposed pairing logic is consistent with the attached `getpairs.for` source, with one important implementation requirement: a KD-tree search must **not** be accepted at face value at the radius boundary or for closest-mode ties. The Python engine therefore uses `cKDTree` only to find candidates, then re-evaluates exact squared distances in float64 and restores the original dataset-2 ordering/tie rule.

The application preserves the algorithmic behavior that matters to the study:

- dataset 1 is the **reference** and drives the outer loop;
- dataset 2 is searched and plotted on X;
- strict radius: `dis < dismax**2`;
- all neighbors or closest neighbor per reference;
- closest ties resolved by first dataset-2 occurrence;
- no one-to-one assignment constraint;
- shared-axis coordinate collapse when an axis is disabled in either dataset;
- output records concatenate all dataset-1 variables followed by all dataset-2 variables.

## 2. FORTRAN-to-Python mapping

| Legacy FORTRAN block | Behavior | Python counterpart |
|---|---|---|
| Parameter-file scan to `START OF PARAMETERS` | Reads 7 parameters in fixed order | `src/core/config.py::parse_getpairs_par` |
| `chknam` | Removes leading blanks / inline filename comments | `_legacy_filename` |
| GSLIB header reads | title, `nvari`, names, data | `src/core/io.py::read_gslib_bytes` |
| `if(icx1 > 0 .and. icx2 > 0)` etc. | Shared-axis population quirk | `src/core/pairing.py::prepare_coordinate_arrays` |
| `dsqd = dismax**2` | Squared search radius | `pair_records_kdtree` and `brute_force_pairs` |
| Nested `do j=1,n1`, `do i=1,n2` | Reference-driven ordering | `brute_force_pairs`; KD output reordered identically |
| `if(dis < dsqd)` | Strict radius | post-KD exact squared-distance filter |
| `ikeepclose == 0` | Emit every pair inside radius | `_pair_all` using `query_ball_point` |
| `dis < dclose` | Closest only; first tie retained | `_pair_closest` using `query` plus exact tie check |
| `pdat(iclose)` warning | Reuse is allowed | `pairing_report` counts reused comparison records |
| `write(lout,...) d1(j), d2(i)` | Full records concatenated | `paired_to_gslib_bytes` |
| commented `tmin` checks | No active trimming in legacy code | explicit Streamlit assay-validity/trimming controls |

## 3. Validation of the requested algorithm

### 3.1 Reference direction

Correct. In the source, `j=1,n1` is the outer loop. Dataset 1 therefore defines the population for the pairing rate and unpaired-reference count. It is correct to plot dataset 1 on Y and dataset 2 on X when reproducing the study figure.

### 3.2 Strict maximum distance

Correct and important. The source compares squared distance using:

```text
if(dis < dsqd)
```

not `<=`. Consequently, a sample exactly 2.000 m away is excluded for a 2.000 m search radius. `cKDTree.query_ball_point` may return radius-boundary candidates, so the Python code always rechecks `d2 < dismax**2` after the tree search.

### 3.3 All-pairs mode

Correct. A single reference record can generate multiple output rows. To reproduce the exact legacy row order, KD-tree neighbor indices are sorted into original dataset-2 row order before output.

### 3.4 Closest-only mode

Correct. The source initializes `dclose=dsqd` and updates only for `dis < dclose`. Therefore:

- a boundary-distance record is never selected;
- the first record at the minimum distance wins an exact tie;
- there is at most one output row per reference record.

`cKDTree.query()` is fast but does not guarantee the same tie index as the FORTRAN loop. The Python implementation uses `query()` as the nearest-neighbor seed, finds all exact-distance tie candidates around that radius, computes squared distances again, and selects the smallest original dataset-2 row index among exact minima.

### 3.5 Dataset-2 reuse

Correct. GETPAIRS does not enforce a 1:1 assignment. The `pdat` array only prints a warning when a comparison point is selected again in closest mode. The Python application preserves reuse and reports:

- number of comparison records used more than once;
- total extra reuse assignments.

### 3.6 2D behavior

Correct. The coordinate logic is not independent by file. For example, if dataset 1 has a Z column but dataset 2 sets Z=0 in the parameter file, **both** Z arrays remain 0.0. This converts the distance calculation to XY only. The app preserves the general rule and exposes the common operation explicitly as **2D search (ignore Z)**.

### 3.7 Valid-assay filtering

The study description requires only samples with valid assay results for the evaluated variable. This is not active in `getpairs.for` because the `tmin` statements are commented out. Therefore the study workflow necessarily includes preprocessing outside the live legacy code or in a modified workflow. The app makes this step explicit and applies it before pairing by default, which is necessary if Stot, S2, and Au are expected to have different pair counts.

## 4. Deliberate deviations and modernizations

1. **KD-tree acceleration.** The O(n1×n2) nested search is replaced by `scipy.spatial.cKDTree`, while strict post-filtering and output-order restoration preserve the logical pair set.
2. **Float64 arithmetic.** The Python code performs internal calculations in 64-bit floating point, as requested. The original FORTRAN declares default `REAL`, typically single precision on conventional compilers. Results can theoretically differ for pathological points lying within machine precision of the radius boundary; the logical predicate itself is unchanged.
3. **Explicit assay/null/trimming controls.** The legacy `tmin` code is commented out. The app exposes valid-assay filtering and optional grade-range trimming as deliberate preprocessing controls.
4. **Reuse reporting instead of console warnings.** Dataset-2 reuse remains allowed. The app quantifies reuse rather than emitting one warning line per event.
5. **Additional input formats.** CSV and Excel are accepted in addition to mandatory GSLIB support.
6. **Validation and diagnostics.** Malformed GSLIB headers, duplicated columns, non-numeric coordinates, empty filtered datasets, empty pair sets, and near-zero percentage denominators are explicitly trapped.
7. **Modern visualization.** Plotly replaces the static plotting workflow and adds collision-aware metric annotation, display-only clipping, and export controls. Statistics remain based on the un-clipped paired values.

The following are **not deviations**: reference-driven looping, strict radius, all/closest modes, first-occurrence tie logic, and allowing dataset-2 reuse.

## 5. Pairing-rate definition

For all-pairs mode, `number_of_pairs / number_of_reference_records` can exceed 100% and is therefore not a meaningful pairing rate. The application defines:

```text
Pairing rate (%) = unique paired reference records / eligible reference records × 100
```

This remains bounded by 100% and directly measures reference-sample coverage.

## 6. Multiplot interpretation and one source inconsistency

The figure is reproduced in the same panel order:

1. top-left density scatter, comparison X vs reference Y;
2. top-right empirical CDFs;
3. bottom-left Q-Q;
4. bottom-right statistics table.

The requested `c` annotation is implemented as the Reduced Major Axis slope:

```text
c = sign(Pearson r) × stdev(reference) / stdev(comparison)
```

However, the Stot source figure reports `c = 1.16` while the displayed table reports stdev values 1.96 and 1.80. Those rounded values imply an RMA slope near 1.09, not 1.16. Therefore the source figure's `c` is not numerically consistent with the RMA definition supplied in the new specification (or it was computed from a different/unrounded dataset or formula). The Python app follows the explicit specification rather than forcing the displayed source value.

## 7. Statistics and denominator guard

The study-order table uses:

`count, mean, stdev, cv, min, P10, P50, P90, max`

with sample standard deviation (`ddof=1`) and `cv = stdev / mean`.

The difference is calculated at full precision:

```text
Diff.(%) = (comparison - reference) / reference × 100
```

A near-zero guard is based on a configurable fraction of the reference variable characteristic scale, defined as `max(|mean|, stdev, machine epsilon)`; CV uses a dimensionless scale of 1. The default of 0.001 flags a Stot minimum near 0.005 when the mean is about 7, preventing a visually dramatic but practically unstable ~5,960% difference.

## 8. Verification tests

`tests/test_pairing.py` compares the cKDTree implementation against the literal brute-force implementation for:

- all-pairs output;
- strict exclusion at exactly `dismax`;
- closest-only output;
- equal-distance tie resolved by first comparison occurrence;
- comparison-record reuse.

`tests/test_coordinate_quirk.py` verifies that disabling Z in only one dataset forces Z=0 in both and therefore produces a 2D match.

Run:

```bash
pytest -q
```

## 9. Direct legacy-output cross-check

A direct execution cross-check was also performed on the supplied FORTRAN source using the synthetic GSLIB datasets. Current `gfortran` does not accept the legacy dummy-character declaration `character str*len`, so the validation copy was changed only syntactically to `character(len=len) :: str`; the pairing logic was left unchanged.

For `dismax = 2.0` and full XYZ search, the legacy and Python implementations produced the same ordered sample-ID pairs:

**All pairs**

```text
(1,101), (1,102), (2,101), (2,102), (2,103), (3,104), (4,105)
```

**Closest only**

```text
(1,101), (2,101), (3,104), (4,105)
```

The legacy run also emitted its multiple-assignment warning for comparison sample 101, while the Python report counts that reuse instead of rejecting it. This direct cross-check supports the unit-test result that the KD-tree implementation reproduces the intended FORTRAN pair set and ordering on the validation case.
## 8. Application architecture

Version 3 uses a standard layered Streamlit application structure. `app.py` is intentionally minimal and only configures the Streamlit page, applies the theme, and invokes the UI page. Analytical code is isolated under `src/core/`; presentation code is isolated under `src/ui/`. This keeps pairing, statistics and I/O independently testable and prevents CSS/widget changes from altering analytical formulas.

The UI layer contains the premium corporate theme, reusable KPI/header/sidebar components and scientific palette definitions. The figure statistics order exposed in Version 2 has been removed from the interface; the study table order is now fixed in the workflow to preserve consistency with the Pueblo Viejo comparison figure.

The visible pairing labels were also changed from the implementation-oriented `All pairs` / `Closest only` wording to:

- `All neighbors within radius` = legacy `ikeepclose = 0`;
- `Nearest neighbor only` = legacy `ikeepclose = 1`.

The underlying algorithm is unchanged.



## 10. Spatial pair-location visualization

Version 4 adds a spatial QA/QC view above the statistical multiplot. The plot uses exactly the `analysis_mask` pair rows that feed the selected-variable statistics and the comparison figure; it does not display unpaired records or spatial pairs subsequently excluded for invalid selected-variable values.

The default projection is **XY Plan View**. **XZ** and **YZ** section views are offered only when both datasets contain valid mapped coordinates for every displayed pair in the required axes. Each pair is drawn as a low-opacity connector between the reference and comparison sample locations, with corporate-colour markers for the two series. Hover text reports the pair number, sample IDs (when mapped), separation distance, and selected-variable value.

The visualization uses the original mapped coordinates rather than the coordinate arrays after legacy axis collapse. Therefore, when the user selects **2D search (ignore Z)**, XZ/YZ can still be used for geological/spatial context if valid Z values exist; the UI explicitly notes that Z did not participate in pair acceptance or separation calculations. This separation between **search coordinates** and **display coordinates** prevents the section view from falsely implying that Z controlled a 2D pairing result.
