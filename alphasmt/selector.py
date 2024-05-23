from alphasmt.utils import parNReward
from alphasmt.parser import s1_strat_parse, SOLVER_TACTICS, PREPROCESS_TACTICS

# no incentive for early termination if not solved for now
# return True if res2 is better than res1
def compare_performance(res1, res2):
    if not res2[0]:
        return False
    if not res1[0]:
        return True
    return res2[1] < res1[1]
    
def virtual_add_strategy(cur_best_res, strat_res):
    assert (len(cur_best_res) == len(strat_res))
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
def linear_strategy_select(max_selected, result_database, timeout):
    assert (max_selected <= len(result_database))
    selected_strat = []
    log_str = "\n"
    best_res = [(False, None)] * len(result_database[list(result_database.keys())[0]])
    for i in range(max_selected):
        best_strat = None
        best_value = parNReward(best_res, N, timeout)
        for strat in result_database:
            if strat in selected_strat:
                continue
            strat_res = result_database[strat]
            virtual_res, _ = virtual_add_strategy(best_res, strat_res)
            virtual_value = parNReward(virtual_res, N, timeout)
            if virtual_value > best_value:
                best_strat = strat
                best_value = virtual_value
        if best_strat is None:
            break
        selected_strat.append(best_strat)
        best_res, beat_num = virtual_add_strategy(best_res, result_database[best_strat])
        log_str += f"Select {i}'th strategy: {best_strat}\nimprove on {beat_num} instances,  new par10: {best_value:.5f}\n"
    return selected_strat, log_str

def convert_strats_to_act_lists(s1_strat_lst):
    tactic_dict = {}
    solverID = 1000
    act2solver = {}
    preprocessID = 2000
    act2preprocess = {}
    strat_act_lst = []
    s1strat2acts = {}

    for s1_strat in s1_strat_lst:
        tac_lst = s1_strat_parse(s1_strat) 
        act_lst = []
        for tac in tac_lst:
            tac_str = str(tac) # currently it is not guaranteed that the parameters are in the same order after parsing
            if tac_str not in tactic_dict:
                if tac[0] in SOLVER_TACTICS:
                    tactic_dict[tac_str] = solverID
                    act2solver[solverID] = tac
                    solverID += 1
                elif tac[0] in PREPROCESS_TACTICS:
                    tactic_dict[tac_str] = preprocessID
                    act2preprocess[preprocessID] = tac
                    preprocessID += 1
                else:
                    raise Exception(f"Unknown tactic {tac}")
            assert(tac_str in tactic_dict)
            act_lst.append(tactic_dict[tac_str])
        strat_act_lst.append(act_lst)
        s1strat2acts[s1_strat] = tuple(act_lst)

    return strat_act_lst, act2solver, act2preprocess, s1strat2acts

# check whether lst1 is a strick prefix of lst2
def is_strick_prefix(lst1, lst2):
    if len(lst1) >= len(lst2):
        return False
    for i in range(len(lst1)):
        if lst1[i] != lst2[i]:
            return False
    return True
    
def search_next_action(cur_act_path, strat_act_lst):
    action_set = set()
    for strategy in strat_act_lst:
        if is_strick_prefix(cur_act_path, strategy):
            action_set.add(strategy[len(cur_act_path)])
    return list(action_set)
    
            
                
            

    


    