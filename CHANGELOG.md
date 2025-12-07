# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-07

- Implemented IncEnu algorithm restricted to polyhedral regions.
- Implemented delition-restriction algorithm for polyhedral regions.
- Added `volume.py` module for volume computation of polytopes.
- Implemented poisson zero-cell polytope sampling.
- Changed rational arithmetic from `fractions.Fraction` to `gmpy2.mpq`.
- Added predifined polytopes and arrangements, for easy experimentation.
- Added a new strategy "v-entropy" for hyperplane selection in partitioning.
- Added `stats` function to `PartitionTree` for gathering tree statistics.

## [0.1.1] - 2025-09-23

- Added changelog file.
- Minor documentation updates.
- Fixed computation of average depth in partition tree.

## [0.1.0] - 2025-09-23

- Initial release.
