# Logic configs

One JSON file per SMT-LIB logic (e.g. `QF_NIA.json`), loaded by
[`z3alpha.tactics.logic_config.load_logic_config`](../logic_config.py). Each
file defines the tactic action space (and optional parameter search grid)
that stage-1 MCTS is allowed to pick from for that logic.

`default.json` is not an automatic fallback — it's only used if a caller
explicitly requests `logic="default"`.

Tactic names, param names, and `"default"` values below are validated
against **z3 4.16.0**. If the project's z3 version changes, re-run the
validator (see below) — defaults and accepted params can change between
versions.

## Format

Top-level keys are z3 tactic names (must resolve via
[`z3alpha.tactics.catalog.NAME_TO_ID`](../catalog.py), and must be real z3
tactics). Each entry is:

```json
"simplify": {
  "solver": false,
  "params": {
    "elim_and": { "default": "false", "values": ["true", "false"] }
  }
}
```

- `"solver"` (optional, default `false`): `true` if this tactic terminates a
  strategy (goes in `solver_tactics`); `false` if it's a preprocessing step
  that hands off to another tactic (goes in `preprocess_tactics`).
- `"params"` (optional, default `{}`): a search grid for `using-params`.
  Each param maps to `{"default": ..., "values": [...]}`, where `"default"`
  is z3's own default for that param (used as a baseline/no-op choice) and
  `"values"` is the grid of values to try instead. Bool-valued params use
  the strings `"true"`/`"false"` (not JSON booleans), since they're rendered
  verbatim into the `(using-params ...)` tactic string.

A tactic with no params at all (e.g. `"ctx-simplify": {}`) is just applied
by name, with no `using-params` wrapper.

The `"params"` grids are consumed by the UCB1 MAB parameter search during
stage-1 MCTS (see [`z3alpha/mcts/param_selection.py`](../../mcts/param_selection.py)).

## Keeping this in sync with z3

[`scripts/validate_logic_configs.py`](../../../scripts/validate_logic_configs.py)
checks every tactic name, param name, and declared `"default"` against the
z3 binary itself:

```sh
python scripts/validate_logic_configs.py            # check
python scripts/validate_logic_configs.py --fix       # auto-correct defaults
```

Run this after bumping the z3 version, or after editing any of these files.
