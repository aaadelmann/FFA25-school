# FFA school space-charge figures

The zero-current run completed 128 turns in 7 h 23 min. Its updated NAFF
estimates use 127 complete turns, azimuth 0.25–127.25. The three charged cases
retain their original first 15-turn preliminary estimates; none has a completed
128-turn record at this update. All use the same 768 corresponding central-slice
source rows in 98,304-particle, 20 cm KV bunches, with separately matched
covariances at fixed emittances. Currents: 0, 4.748770, 8.701339, 10.399311 mA.
The zero solver is NONE; charged solvers use FFT2D5 and PIPEMODE=NONE
(longitudinal self-field disabled).

The plots label mixed record lengths. Each method subtracts its own zero-current
median; NAFF now uses the 127-turn zero baseline, including for the unchanged
15-turn charged estimates. The common 15-turn CSV snapshot is retained in
preliminary-15turn/ for comparisons with identical record lengths.

The completed zero source was checked in all 8,212 HDF5 frames: all 98,304 IDs,
finite positions/momenta, exact zero charge, source file unchanged during read.
The maximum all-particle kinetic-energy difference from 12 MeV is 1.4133e-8 MeV.
The first 15 turns reprocess within 2.85e-14 of the previous tunes. First/last
63-turn per-particle tune differences at the 95th percentile are 1.8e-5 horizontal
and 8.8e-5 vertical. Completion does not itself establish convergence at 1e-6.
Audit and analysis-checks JSON files retain the checks. Source scripts and raw
selected coordinates: OPALX output/ffa-sc-lumi-2026-11-13/full-record-20261114/.

Files:
- particle-tunes.csv: per-particle NAFF, raw/centroid-subtracted estimates,
  record-length checks, and independent analytic KV predictions.
- summary.csv: median and 5–95% tune ranges and comparison metrics.
- initial-action-fractions.csv: initial horizontal normalized-action fractions
  recovered from actual launch-file second moments, not inferred from tunes.
- tune-shifts-from-zero.csv: difference of each method's medians from its own
  zero-current baseline. This differs slightly from median paired-particle
  differences in summary.csv.

Recreate figures using the OPALX Python environment:

    ~/.venv-h6/bin/python plot_footprint.py
    ~/.venv-h6/bin/python plot_tune_shifts.py

Outputs are written to ../figures/. The colour scale is fixed to [0,1] and
means initial normalized horizontal action divided by the sum of normalized
horizontal and vertical actions. The metric uses each current's matched
covariance. The recovered KV shell squared radius equals 4; corresponding
colour labels across the four actual sources agree within 2.0e-12.
KV crosses are independent model predictions and do not use the colour scale.
Shift bars show shifted particle percentiles, not statistical uncertainty.

Model/source checks used when drafting Section 3.1:
- Independent KV model: OPALX checkout output/ffa-sc-lumi-2026-11-13/kv_optics.py;
  native zero-current tune checks agree within 3.2e-11 horizontally and
  1.2e-8 vertically; matched covariance and integration-resolution checks retained.
- FFT2D5: src/SpaceCharge/FFT2D5/FFT2D5Algorithm.hpp (CIC density, field gathering,
  magnetic self-field, PIPEMODE NONE gradient zeroing), FFT2D5Poisson.cpp
  (free-space convolution and finite differences), FFT2D5FieldStorage.cpp
  (transverse widths and full-reference-path longitudinal spacing).
- Extraction and NAFF scripts/provenance: OPALX checkout
  output/ffa-sc-lumi-2026-11-13/preliminary/. Selected completed HDF5 frames were
  read without changing any simulation; storage order was remapped by persistent
  IDs in every frame, all 98,304 IDs checked, selected states finite. Five newest
  frames excluded; fresh read-only handles avoid stale metadata while files grow.
- LUMI campaign:
  /flash/project_465002453/school-16-cell-sc/current-scan-20261113.

The four-current tune-footprint.pdf is also mirrored at the previously shared
OPALX output/ffa-kv-zero-charge-2026-11-13/tune-footprint.pdf. The earlier
zero-current-only view is retained there as tune-footprint-zero-current.pdf;
its historical report.py now regenerates that separate name.

Build the notes from the parent tex directory:

    pdflatex -interaction=nonstopmode -halt-on-error school-machines.tex
    pdflatex -interaction=nonstopmode -halt-on-error school-machines.tex

## Native one-turn matrix

The new subsection immediately after matching is numbered 3.1.5 in this edition.
It uses the native COF JSON from the verified 12 MeV machine, not the KV
self-field matrix. Files one-turn-matrix-cof-refined.json and
one-turn-matrix-cof-default.json retain the full-precision matrices and metadata.
Refinement changes DT from 8 to 4 ns and halves the finite-difference position
step from 10 to 5 micrometres (momentum steps are those values times P0=beta*gamma).
These coordinates are section positions in metres and mechanical momenta divided
by mc. They are not slopes or the moving orbit-normal coordinates.

    ~/.venv-h6/bin/python check_one_turn_matrix.py

This verifies every displayed matrix entry against rounded native JSON,
reproduces block determinants and fractional/full tune consistency, and writes
one-turn-matrix-checks.json. The implementation description was checked against
OPALX src/Algorithms/ClosedOrbitSolver.cpp, src/Algorithms/OneTurnMap.cpp,
src/Algorithms/ExternalFieldRayTracker.cpp and src/Track/CofCmd.cpp. COF here uses
DOP853 and external fields only; the collective beam study uses Boris plus
FFT2D5. No new tracking was needed to document these retained native results.
