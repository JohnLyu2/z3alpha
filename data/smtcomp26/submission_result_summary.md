# Z3-alpha2 Submission Summary

The results below are based on a timeout of 10min.

| Division | Logic | Portfolio size | Instances | Baseline solved | z3alpha2 solved | Δ |
|----------|-------|---------------:|----------:|----------------:|----------------:|--:|
| Arith | LIA | 1 | 300 | 233 | 239 | +6 (+2.6%) |
| Arith | LRA | 6 | 1,013 | 902 | 934 | +32 (+3.5%) |
| Arith | NIA | 2 | 254 | 224 | 235 | +11 (+4.9%) |
| Arith | NRA | 2 | 99 | 93 | 95 | +2 (+2.2%) |
| QF_Datatypes | QF_DT | 4 | 352 | 142 | 183 | +41 (+28.9%) |
| QF_Datatypes | QF_UFDT | 4 | 200 | 63 | 134 | +71 (+112.7%) |
| QF_NonLinearRealArith | QF_NRA | 5 | 909 | 565 | 693 | +128 (+22.7%) |
| QF_NonLinearIntArith | QF_NIA | 6 | 6,130 | 2,917 | 4,092 | +1,175 (+40.3%) |
| QF_BitVec | QF_BV | 7 | 4,525 | 2,982 | 3,568 | +586 (+19.7%) |
| QF_LinearIntArith | QF_LIA | 6 | 1,239 | 782 | 1,003 | +221 (+28.3%) |
| QF_LinearIntArith | QF_IDL | 6 | 480 | 153 | 240 | +87 (+56.9%) |
| QF_Equality_NonLinearArith | QF_ANIA | 2 | 155 | 92 | 108 | +16 (+17.4%) |
| QF_Equality_NonLinearArith | QF_UFDTNIA | 4 | 80 | 59 | 66 | +7 (+11.9%) |
| QF_Equality_NonLinearArith | QF_UFNIA | 6 | 339 | 193 | 226 | +33 (+17.1%) |
| QF_Equality_NonLinearArith | QF_UFNRA | 2 | 48 | 45 | 48 | +3 (+6.7%) |

* Z3-alpha2 first uses a mix of MCTS and LLM-guidance to explore high-performing, complementary individual strategies, and then combine them using SMT-Select.