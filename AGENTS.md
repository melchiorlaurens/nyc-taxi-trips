# Agent Notes

- Keep logic simple: do not add fallback chains or heuristic path guessing unless explicitly requested.
- Update shared paths/defaults via `src/utils/paths.py`.
- Use `src/utils/common_functions.py` for shapefile → GeoJSON helpers and keep transformations shared.
- `src/utils/get_data.py` handles parquet + zone asset downloads (shapefile zip + lookup CSV).
- Tests live under `tests/`; run `python3 -m pytest` after installing deps from `requirements.txt`.
- Prefer `python3` over `python` in commands unless you know the environment provides the alias.
- Write in README only when prompted.
