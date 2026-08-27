# Technical Note — Pairing Algorithm Based on FORTRAN GETPAIRS

## 1. Purpose

The application performs spatial pairing between two sample datasets using the logic of the FORTRAN **GETPAIRS** program.

The objective is simple:

- **Dataset 1 = Reference**
- **Dataset 2 = Comparison**
- For each reference sample, search Dataset 2 for samples located inside a user-defined maximum distance.
- Retain either all valid neighbors or only the closest neighbor, depending on the selected pairing rule.

The resulting pairs are then used for the statistical comparison.

In the current application workflow, the normal default is:

```text
3D XYZ spherical search
Nearest neighbor only
```

unless an imported `getpairs.par` specifies otherwise.

---

## 2. Core pairing logic

For each reference sample `j`, the algorithm evaluates candidate comparison samples `i`.

The squared separation distance is:

```text
dis = (Xref - Xcmp)^2
    + (Yref - Ycmp)^2
    + (Zref - Zcmp)^2
```

The search threshold is:

```text
dsqd = dismax^2
```

A pair is accepted only when:

```text
dis < dsqd
```

The inequality is **strict**.

Therefore, a comparison sample located exactly at the selected search distance is excluded.

Example:

```text
dismax = 2.0 m

distance = 1.99 m  -> accepted
distance = 2.00 m  -> excluded
distance = 2.01 m  -> excluded
```

---

## 3. Pairing workflow

```mermaid
flowchart TD
    A[Load Reference and Comparison datasets] --> B[Apply active eligibility filters]
    B --> C[Build search coordinates X Y Z]
    C --> D[Select one Reference sample]
    D --> E[Find Comparison candidates near Reference]
    E --> F{distance < dismax?}

    F -- No --> G[Reject candidate]
    F -- Yes --> H{Pairing rule}

    H -- All neighbors within radius --> I[Retain every valid candidate]
    H -- Nearest neighbor only --> J[Retain closest valid candidate]

    I --> K[Write paired Reference + Comparison record]
    J --> K

    K --> L{More Reference samples?}
    G --> L

    L -- Yes --> D
    L -- No --> M[Final paired dataset]
    M --> N[Statistics, plots and sensitivity diagnostics]
```

The reference dataset always drives the search. This means the algorithm is **reference-centered**, not a symmetric nearest-neighbor matching procedure.

---

## 4. Pairing modes

### 4.1 All neighbors within radius

This corresponds to:

```text
ikeepclose = 0
```

Every comparison sample satisfying the strict distance criterion is retained.

Example:

```text
Search radius = 2.0 m

Reference R1
   |
   |-- C1 = 0.8 m  -> retain
   |-- C2 = 1.4 m  -> retain
   |-- C3 = 2.5 m  -> reject
```

Output:

```text
R1 - C1
R1 - C2
```

A single reference sample can therefore generate multiple paired rows.

---

### 4.2 Nearest neighbor only

This corresponds to:

```text
ikeepclose = 1
```

Only the closest comparison sample inside the search radius is retained.

Using the same example:

```text
Reference R1
   |
   |-- C1 = 0.8 m  -> closest
   |-- C2 = 1.4 m
   |-- C3 = 2.5 m  -> outside radius
```

Output:

```text
R1 - C1
```

There is at most one retained comparison sample for each reference sample.

This is the normal default pairing mode in the application when no imported parameter file overrides it.

---

## 5. Equal-distance ties

In nearest-neighbor mode, the closest candidate is updated only when a new distance is strictly smaller than the current closest distance.

Conceptually:

```text
if dis < dclose:
    dclose = dis
    keep candidate
```

Because the comparison is `<` rather than `<=`, two candidates at exactly the same minimum distance do not replace one another.

The first comparison record encountered at that minimum distance is retained.

Example:

```text
R1 -> C10 = 1.25 m
R1 -> C11 = 1.25 m
```

Result:

```text
R1 - C10
```

assuming `C10` occurs first in Dataset 2.

---

## 6. Comparison-sample reuse

The method does **not** impose a global one-to-one assignment.

A comparison sample can be paired with more than one reference sample.

Example:

```text
R1 ----        >---- C5
R2 ----/
```

This is valid.

The search is performed independently for each reference sample, so selecting `C5` for `R1` does not remove `C5` from consideration for `R2`.

---

## 7. 3D and Plan-view search

### 7.1 3D XYZ spherical search

This is the normal search geometry.

When X, Y, and Z are active:

```text
dis = dx^2 + dy^2 + dz^2
```

The search neighborhood is spherical when the same maximum distance is applied in all three directions.

A sample must satisfy the distance threshold in full three-dimensional space.

---

### 7.2 Plan-view search (XY, ignore Z)

Plan-view search is an advanced option.

When enabled:

```text
Zref = 0
Zcmp = 0
```

and therefore:

```text
dis = dx^2 + dy^2
```

The search becomes circular in plan view.

Because vertical separation is ignored:

```text
distance_2D <= distance_3D
```

for the same two samples.

Therefore, Plan-view search can produce **more pairs than 3D search**, not fewer.

A pair that fails the 3D radius because of vertical separation can still pass the XY radius.

The spatial visualization may still display original Z values for geological context, but Z does not contribute to pair acceptance when Plan-view search is active.

---

## 8. Eligibility filtering before pairing

The application reduces the eligible records before the spatial search.

The sequence is:

```text
Input data
   -> eligibility filters
   -> spatial search
   -> paired dataset
   -> statistics
```

The main eligibility controls are:

```text
Positive assays only (> 0)
Grade range
Categorical variable
```

### Positive assays only (> 0)

This filter is enabled by default for the selected analytical variable.

A record is eligible only when the selected variable is:

```text
finite AND > 0
```

This removes values such as:

```text
-99
-999
0
NaN
Inf
```

before pairing.

This is an application-level preprocessing rule. It is **not** part of the original FORTRAN GETPAIRS distance algorithm.

The user may disable it when a variable legitimately contains zero or negative values.

### Grade-range filter

An optional minimum and maximum value can be applied before pairing.

### Categorical Variable Filter

The categorical filter is presented as a primary filter.

Its default state is:

```text
<No filter>
```

so no categorical restriction is applied until the user explicitly selects a field and category values.

Typical categorical fields include:

```text
Destination
MetType
Lithology
Alteration
Domain
Weathering
Sample type
Year
```

The corresponding categorical field can be selected independently in the two input datasets.

Only samples that pass all active eligibility filters are available to become pairs.

---

## 9. Python implementation

A literal nested search between every reference and comparison sample would require approximately:

```text
Nreference × Ncomparison
```

distance evaluations.

The application accelerates candidate identification using:

```text
scipy.spatial.cKDTree
```

The KD-tree is used only to identify nearby candidates efficiently.

The final acceptance logic still applies the GETPAIRS condition explicitly:

```text
distance_squared < dismax^2
```

This preserves the important behavior of the original algorithm while providing much better performance for large datasets.

---

## 10. Pairing Sensitivity

The application includes a **Pairing Sensitivity** diagnostic to evaluate the robustness of the paired population to the selected search radius.

The main pairing result is calculated using the active user-selected radius.

The sensitivity diagnostic then repeats the same pairing logic for a sequence of alternative radii while preserving:

```text
Reference dataset
Comparison dataset
Search geometry
Pairing mode
Eligibility filters
```

A standard sequence is approximately:

```text
1, 2, 3, 4, 5 m
```

with the active selected radius also included when necessary.

The diagnostic can summarize:

```text
Number of pairs
Pairing rate
Pair-separation behavior
```

This procedure is analogous to testing the search-radius sensitivity of the pairing configuration.

It does **not** alter the main paired dataset and it does not introduce a different matching algorithm.

---

## 11. Final paired dataset

Each accepted pair contains:

```text
Reference record
+
Comparison record
+
Pair separation distance
```

Conceptually:

```text
Reference R1  <---- 1.23 m ---->  Comparison C8
Reference R2  <---- 0.74 m ---->  Comparison C5
Reference R3  <---- 1.91 m ---->  Comparison C5
```

These exact paired records are the population used by the subsequent statistical comparison and spatial pair-location plots.

---

## 12. Algorithm summary

The complete logic can be summarized as:

```text
Apply active eligibility filters.

FOR each Reference sample:

    identify nearby Comparison candidates

    FOR each candidate:

        calculate squared spatial distance

        IF distance_squared < dismax^2:

            IF mode = all neighbors:
                retain candidate

            IF mode = nearest neighbor:
                retain only the closest candidate
                preserve first occurrence in an exact tie

Comparison records remain reusable.

Return the final paired dataset.

Optionally repeat the same logic at alternative search radii
for Pairing Sensitivity diagnostics.
```

The key characteristics are therefore:

- reference-driven search;
- strict distance threshold;
- normal default of 3D XYZ spherical search;
- all-neighbor or nearest-neighbor mode;
- nearest-neighbor mode as the normal application default;
- first-occurrence tie handling;
- comparison-record reuse;
- advanced Plan-view search that ignores Z;
- positive-assay preprocessing enabled by default;
- visible categorical filtering with `<No filter>` as the default;
- eligibility filtering before pairing;
- KD-tree acceleration without changing the pairing rule;
- search-radius sensitivity diagnostics that do not alter the main pairing result.
