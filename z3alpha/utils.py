def solved_num(res_list):
    return len([res for res in res_list if res[0]])


def solved_num_reward(res_list):
    return solved_num(res_list) / len(res_list)


def par_n(res_list, n, timeout):
    par_n_value = 0
    for i in range(len(res_list)):
        if not res_list[i][0]:
            par_n_value += n * timeout
        else:
            par_n_value += res_list[i][1]
    return par_n_value


def par_n_reward(res_list, n, timeout):
    par_n_value = par_n(res_list, n, timeout)
    max_par_n = len(res_list) * timeout * n
    return 1 - par_n_value / max_par_n


def calculate_percentile(values, percentile):
    assert len(values) > 0
    # percentile is of the form like "90p"
    percent = float(percentile[:-1]) / 100
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percent)
    return sorted_values[index]


def encode_strat_row(strat, res):
    """Encode one (strat, per-bench-results) pair as a CSV row.

    Each result tuple is ``(solved, time, ...)``. Solved instances keep their
    time, unsolved ones are written as the negation so the sign tracks
    solved/unsolved without needing a separate column.
    """
    return [strat] + [r[1] if r[0] else -r[1] for r in res]


def reward_dispatcher(timeout):
    """Map ``reward_type`` name -> reward function over a per-benchmark result list.

    Used by both linear (``LinearStrategyGame``) and stage-2 (``Stage2StrategyGame``)
    environments so the set of supported reward types stays in one place.
    """
    return {
        "#solved": solved_num_reward,
        "par2": lambda results: par_n_reward(results, 2, timeout),
        "par10": lambda results: par_n_reward(results, 10, timeout),
    }
