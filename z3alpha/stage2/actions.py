from z3alpha.parser import PREPROCESS_TACTICS, SOLVER_TACTICS, s1_strat_parse


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
            tac_str = str(
                tac
            )  # currently it is not guaranteed that the parameters are in the same order after parsing
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
            assert tac_str in tactic_dict
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
