def solve_with_cache(
    bench_id: int,
    linear_strategies: list[tuple[list[int], int]],
    cache_database: dict[tuple[int, ...], list[tuple[bool, float, str]]],
    timeout: int,
) -> tuple[bool, float]:
    time_remain = timeout
    solved = False
    time_used = 0
    for strategy, strategy_timeout in linear_strategies:
        cache_solved, cache_time, _ = cache_database[tuple(strategy)][bench_id]
        if strategy_timeout < cache_time:
            time_remain -= strategy_timeout
            time_used += strategy_timeout
        elif time_remain < cache_time:
            # Cache says this run exceeds remaining budget; consume all remaining time.
            time_used += time_remain
            time_remain = 0
        else:
            if cache_solved:
                solved = True
                time_used += cache_time
                break
            time_remain -= cache_time
            time_used += cache_time
        if time_used >= timeout:
            break
    return solved, time_used


def evaluate_stage2_with_cache(
    benchmark_count: int,
    get_linear_strategies_for_bench,
    cache_database: dict[tuple[int, ...], list[tuple[bool, float, str]]],
    timeout: int,
) -> list[tuple[bool, float]]:
    results = []
    for bench_id in range(benchmark_count):
        linear_strategies = get_linear_strategies_for_bench(bench_id)
        results.append(
            solve_with_cache(bench_id, linear_strategies, cache_database, timeout)
        )
    return results
