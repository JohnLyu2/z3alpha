from dataclasses import dataclass


@dataclass(frozen=True)
class Stage2Context:
    seed_action_sequences: list[list[int]]
    solver_actions: dict[int, tuple]
    preprocess_actions: dict[int, tuple]
    result_cache: dict[tuple[int, ...], list[tuple[bool, float, str]]]
    probe_stats: dict[str, dict[str, int]]
    probe_records: list[dict[str, int]]
