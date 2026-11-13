# FFA school space-charge figures

These are preliminary results from the first 15 turns, azimuth 0.25–15.25,
for 768 corresponding particles in the central longitudinal slice of each
98,304-particle, 20 cm KV bunch. Currents: 0, 4.748770, 8.701339,
10.399311 mA. Physical launch covariances are matched separately at each
current, with fixed emittances. The zero-current solver is NONE; charged
cases use FFT2D5 with PIPEMODE=NONE (longitudinal self-field disabled).
The original study targets 128 turns; this retained table is a fixed preliminary
snapshot, not a live update.

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
