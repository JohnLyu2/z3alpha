from z3alpha.utils import par_n_reward


# no incentive for early termination if not solved for now
# return True if res2 is better than res1
def compare_performance(res1, res2):
    if not res2[0]:
        return False
    if not res1[0]:
        return True
    return res2[1] < res1[1]


def virtual_add_strategy(cur_best_res, strat_res):
    assert len(cur_best_res) == len(strat_res)
    # make a deep copy of the list cur_best_res
    new_best_res = [None] * len(cur_best_res)
    beat_count = 0
    for i in range(len(cur_best_res)):
        if compare_performance(cur_best_res[i], strat_res[i]):
            new_best_res[i] = strat_res[i]
            beat_count += 1
        else:
            new_best_res[i] = cur_best_res[i]
    return new_best_res, beat_count


# now only use par10 as metric to select strategies
N = 10


def create_greedy_linear_strategy_portfolio(max_selected, result_database, timeout):
    assert max_selected <= len(result_database)
    selected_strat = []
    log_str = "\n"
    best_res = [(False, None)] * len(result_database[list(result_database.keys())[0]])
    for i in range(max_selected):
        best_strat = None
        best_value = par_n_reward(best_res, N, timeout)
        for strat in result_database:
            if strat in selected_strat:
                continue
            strat_res = result_database[strat]
            virtual_res, _ = virtual_add_strategy(best_res, strat_res)
            virtual_value = par_n_reward(virtual_res, N, timeout)
            if virtual_value > best_value:
                best_strat = strat
                best_value = virtual_value
        if best_strat is None:
            break
        selected_strat.append(best_strat)
        best_res, beat_num = virtual_add_strategy(best_res, result_database[best_strat])
        log_str += f"Select {i}'th strategy: {best_strat}\nimprove on {beat_num} instances,  new par10: {best_value:.5f}\n"
    return selected_strat, log_str
