import copy

# qf_lia may need to add some to simplify and ctx-simplify
BASIC_PARAMS = {
    # "smt"
    10: {
        "random_seed": [i * 100 for i in range(6)]
    },
    # "simplify"
    20: { 
        "elim_and": ["true","false"], # z3-qfbv z3-qfnia z3-qfnra-nlsat z3-qflia z3-qflra fast-qfnia fast-qfbv fast-qfnra
        "blast_distinct": ["true","false"], # z3-qfbv z3-qfnia z3-qfnra-nlsat z3-qflia z3-qflra fast-qfnia fast-qfbv fast-qfnra
        "local_ctx": ["true","false"], # z3-qfbv z3-qfnia z3-qflia fast-qfnia fast-qfbv fast-qfnra
        
    },
    # "propagate-values"
    # 21: {
    #     "push_ite_bv": ["true","false"] # z3-qfbv
    # },
    # "nla2bv" 
    5: {
        "nla2bv_max_bv_size": [4, 8, 16, 32, 64, 128]
    },
    # "qfnra-nlsat" 
    11: {
        "inline_vars": ["true","false"],
        "factor": ["true","false"],
        "seed": [i * 100 for i in range(6)]
    },
    # "add-bounds"
    36: {
        "add_bound_lower": [-2, -4, -8, -16, -32],
        "add_bound_upper": [2, 4, 8, 16, 32]
    },
    # "pb2bv"
    # 8: {
    #     "pb2bv_all_clauses_limit": [4, 8, 16, 32, 64],
    # },
    # "lia2pb"
    38: {
        "lia2pb_max_bits": [16, 32, 64, 128],
    },
}

# for now, MCTS_run just has one unchanged Logic setting; include all relevant (even after conversion) parameters for each logic
def create_params_dict(logic):
    params = copy.deepcopy(BASIC_PARAMS)
    if logic == "QF_LIA":
        # "simplify"
        params[20]["som"] = ["true","false"] # z3-qfbv z3-qfnia z3-qflia z3-qflra fast-qfnia fast-qfbv fast-qfnra
        params[20]["flat"] = ["true","false"] # z3-qfbv z3-qfnia z3-qflia fast-qfnia fast-qfbv fast-qfnra
        params[20]["pull_cheap_ite"] = ["true","false"] # z3-qfbv z3-qfnia z3-qflia fast-qfbv
        params[20]["push_ite_arith"] = ["true","false"]
        params[20]["hoist_ite"] = ["true","false"]
        params[20]["arith_lhs"] = ["true","false"]
    elif logic == "QF_LRA":
        # "simplify"
        params[20]["som"] = ["true","false"]
        params[20]["flat"] = ["true","false"]
    elif logic == "QF_NIA":
        # "simplify"
        params[20]["som"] = ["true","false"]
        params[20]["flat"] = ["true","false"]
        params[20]["hi_div0"] = ["true","false"] # z3-qfnia fast-qfnia fast-qfnra
        params[20]["hoist_mul"] = ["true","false"] # z3-qfbv z3-qfnia fast-qfnia fast-qfbv fast-qfnra
        params[20]["pull_cheap_ite"] = ["true","false"]
        # for qfbv
        params[20]["push_ite_bv"] = ["true","false"]
        # "propagate-values"
        params[21] = {}
        params[21]["push_ite_bv"] = ["true","false"]
    elif logic == "QF_NRA":
        # "simplify"
        params[20]["som"] = ["true","false"]
        params[20]["flat"] = ["true","false"]
        params[20]["hi_div0"] = ["true","false"]
        params[20]["hoist_mul"] = ["true","false"]
        # for qfbv
        params[20]["push_ite_bv"] = ["true","false"]
        params[20]["pull_cheap_ite"] = ["true","false"]
        # "propagate-values"
        params[21] = {}
        params[21]["push_ite_bv"] = ["true","false"]
    elif logic == "QF_BV":
        # "simplify"
        params[20]["som"] = ["true","false"]
        params[20]["flat"] = ["true","false"]
        params[20]["push_ite_bv"] = ["true","false"] # z3-qfbv fast-qfbv
        params[20]["hoist_mul"] = ["true","false"]
        params[20]["pull_cheap_ite"] = ["true","false"]
        # "propagate-values"
        params[21] = {}
        params[21]["push_ite_bv"] = ["true","false"]
    elif logic == "QF_S":
        pass
    else:
        raise NotImplementedError(f"Logic {logic} not implemented")
    return params
        
