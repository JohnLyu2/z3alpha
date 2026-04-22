import csv
import logging
import math
import copy
from pathlib import Path
from z3alpha.environment import StrategyGame
from z3alpha.logging_config import attach_file_logger

logger = logging.getLogger(__name__)

INIT_Q = 0
IS_MEAN_EST = False


class MCTSNode:
    def __init__(self, logic_config, is_mean, trace_log, c_ucb, action_history=None):
        self.param_dict = logic_config["params"] if logic_config else {}
        self.isMean = is_mean
        self.c_ucb = c_ucb
        # self.alpha = alpha
        self.visitCount = 0
        self.actionHistory = [] if action_history is None else list(action_history)
        self.valueEst = 0
        self.children = {}
        self.reward = 0  # always 0 for now
        self._setParamMABs()
        self.trace_log = trace_log

    def __str__(self):
        return str(self.actionHistory)

    def isExpanded(self):
        return bool(self.children)

    def hasParamMABs(self):
        if len(self.actionHistory) == 0:
            return False
        lastestAction = self.actionHistory[-1]
        return lastestAction in self.param_dict.keys()

    def _setParamMABs(self):
        if not self.hasParamMABs():
            return
        lastestAction = self.actionHistory[-1]
        self.params = self.param_dict[lastestAction]
        self.MABs = {}
        self.selected = {}
        for param in self.params.keys():
            MABdict = {}
            for paramValue in self.params[param]:
                MABdict[paramValue] = [0, INIT_Q]  # (visit count, q estimation)
            self.MABs[param] = MABdict
            self.selected[param] = None

    # the argument action is only for log
    def _ucb(self, action_pair, action):
        visitCount, qScore = action_pair
        exploreScore = self.c_ucb * math.sqrt(
            math.log(self.visitCount + 1)  # check ucb 1
            / (visitCount + 0.001)
        )
        ucb = qScore + exploreScore
        self.trace_log.debug(
            f"  Value of {action}: Q value: {qScore:.05f}; Exp: {exploreScore:.05f} ({visitCount}/{self.visitCount}); UCB: {ucb:.05f}"
        )
        return ucb

    # rename parameter values; easily confuesed with the value of a node
    def _selectMAB(self, param):
        MABdict = self.MABs[param]
        selected = None
        bestUCB = -1
        for valueCandidate, pair in MABdict.items():
            # if param == "timeout" and valueCandidate >= remain_time:
            #     continue
            ucb = self._ucb(pair, valueCandidate)
            if ucb > bestUCB:
                bestUCB = ucb
                selected = valueCandidate
        assert bestUCB >= 0
        return selected

    def selectMABs(self):
        for param in self.params.keys():
            self.trace_log.debug(f"\n  Select MAB of {param}")
            selectV = self._selectMAB(param)
            self.trace_log.debug(f"  Selected value: {selectV}\n")
            self.selected[param] = selectV
        return self.selected

    def backupMABs(self, reward):
        for param in self.params.keys():
            MABdict = self.MABs[param]
            selectedV = self.selected[param]

            if self.isMean:
                MABdict[selectedV][1] = (
                    MABdict[selectedV][1] * MABdict[selectedV][0] + reward
                ) / (MABdict[selectedV][0] + 1)
            else:
                MABdict[selectedV][1] = max(MABdict[selectedV][1], reward)
            MABdict[selectedV][0] += 1
            self.selected[param] = None

    # def value(self):
    #     if self.visitCount == 0:  # will this be called anytime?
    #         return 0
    #     return max(self.valueLst)


# A MCTS run starting at a particular node as root; this framework only works for deterministric state transition


class MCTS_RUN:
    def __init__(
        self,
        stage,
        config,
        bench_lst,
        logic,
        z3path,
        value_type,
        log_folder,
        batch_size=1,
        root=None,
        logic_config=None,
    ):
        self.stage = stage
        self.z3path = z3path
        self.config = config
        self.logic_config = logic_config
        self.numSimulations = config["sim_num"]
        self.isMean = IS_MEAN_EST
        self.discount = 1  # now set to 1
        self.c_uct = config["c_uct"]
        # self.alpha = alpha
        self.trainingLst = bench_lst
        self.logic = logic
        self.timeout = config["timeout"]
        self.valueType = value_type
        self.batchSize = batch_size
        if self.stage == 1:
            self.c_ucb = config["c_ucb"]
            self.resS1Database = {}
            self._s1_csv_path = Path(log_folder) / "stage1_strategy_results.csv"
            self._written_strats = set()
            with open(self._s1_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["strat"] + bench_lst)
        else:
            self.c_ucb = None
            self.resS1Database = config["s2dict"]["res_cache"]

        self.trace_log = attach_file_logger(
            f"z3alpha.s{self.stage}mcts",
            Path(log_folder) / f"stage{self.stage}_mcts_trace.log",
        )

        if not root:
            root = MCTSNode(self.logic_config, self.isMean, self.trace_log, self.c_ucb)
        self.root = root
        self.bestReward = -1
        self.topStrategies = [None, None, None] # top 3 strategies
        self.topRewards = [-1, -1, -1]

    def _uct(self, childNode, parentNode, action):
        valueScore = childNode.reward + self.discount * childNode.valueEst
        exploreScore = self.c_uct * math.sqrt(
            math.log(parentNode.visitCount) / (childNode.visitCount + 0.001)
        )
        uct = valueScore + exploreScore
        self.trace_log.debug(
            f"  Value of {action}: Q value: {valueScore:.05f}; Exp: {exploreScore:.05f} ({childNode.visitCount}/{parentNode.visitCount}); UCT: {uct:.05f}"
        )
        return uct

    def _select(self):
        searchPath = [self.root]
        node = self.root
        # does not consider the root has MABs
        while node.isExpanded() and not self.env.isTerminal():
            self.trace_log.debug(f"\n  Select at {node}")
            # may add randomness when the UCTs are the same

            # select in the order as in the list if the same UCT values; put more promising/safer actions earlier in legalActions()
            selected = None
            bestUCT = -1
            nextNode = None
            for action, childNode in node.children.items():
                uct = self._uct(childNode, node, action)
                if uct > bestUCT:
                    selected = action
                    bestUCT = uct
                    nextNode = childNode
            assert bestUCT >= 0
            node = nextNode
            self.trace_log.debug(f"  Selected action {selected}")
            # remainTime = self.env.getRemainTime() if selected == 2 else None
            params = node.selectMABs() if node.hasParamMABs() else None
            searchPath.append(node)
            self.env.step(selected, params)
        return node, searchPath

    def _expandNode(self, node, actions, reward):
        node.reward = reward
        for action in actions:
            history = copy.deepcopy(node.actionHistory)
            history.append(action)
            node.children[action] = MCTSNode(
                self.logic_config, self.isMean, self.trace_log, self.c_ucb, history
            )

    def _rollout(self):
        self.env.rollout()

    def _backup(self, searchPath, sim_value):
        value = sim_value
        for node in reversed(searchPath):
            if self.isMean:
                node.valueEst = (node.valueEst * node.visitCount + value) / (
                    node.visitCount + 1
                )
            else:
                node.valueEst = max(node.valueEst, value)
            node.visitCount += 1
            value = node.reward + self.discount * value  # not applicable now
            if node.hasParamMABs():
                node.backupMABs(value)

    def _oneSimulation(self):
        # now does not consider the root is not the game start
        self.env = StrategyGame(
            self.stage,
            self.trainingLst,
            self.logic,
            self.timeout,
            self.config,
            self.batchSize,
            z3path=self.z3path,
            logic_config=self.logic_config,
        )
        selectNode, searchPath = self._select()
        self.trace_log.info("Selected Node: " + str(selectNode))
        self.trace_log.info("Selected Strategy ParseTree: " + str(self.env))
        if self.env.isTerminal():
            self.trace_log.info("Terminal Strategy: no rollout")
            value = self.env.getValue(self.resS1Database, self.valueType)
        else:
            actions = self.env.legalActions()
            # now reward is always 0 at each step
            self._expandNode(selectNode, actions, 0)
            self._rollout()
            self.trace_log.info(f"Rollout Strategy: {self.env}")
            value = self.env.getValue(self.resS1Database, self.valueType)
        self._updateTopStrategies(value, str(self.env))
        self.trace_log.info(f"Final Return: {value}\n")
        self._backup(searchPath, value)

    def _updateTopStrategies(self, value, stratetgy):
        for i in range(3):
            if value > self.topRewards[i]:
                if i == 0:
                    msg = f"At sim {self.num_sim}, new best reward found: {value:.5f}"
                    logger.info(msg)
                    self.trace_log.info(msg)
                self.topRewards.insert(i, value)
                self.topRewards.pop()
                self.topStrategies.insert(i, stratetgy)
                self.topStrategies.pop()
                break

    def _write_new_results(self):
        """Append any newly evaluated strategies to the S1 results CSV."""
        new_strats = set(self.resS1Database.keys()) - self._written_strats
        if not new_strats:
            return
        with open(self._s1_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for strat in new_strats:
                res = self.resS1Database[strat]
                row = [strat] + [r[1] if r[0] else -r[1] for r in res]
                writer.writerow(row)
        self._written_strats.update(new_strats)

    def start(self):
        for i in range(self.numSimulations):
            self.num_sim = i
            # Console: progress for stage 1 only (stage 2 can have huge sim_num).
            if self.stage == 1:
                logger.info(f"Simulation {i} starts")
            self.trace_log.info(f"Simulation {i} starts")
            self._oneSimulation()
            if self.stage == 1:
                self._write_new_results()

    # def bestNS1Strategies(self, n):
    #     if n > len(self.resS1Database):
    #         n = len(self.resS1Database)
    #     return sorted(self.resS1Database, key=self.resS1Database.get, reverse=True)[:n]

    def getStrategyStat(self, strat):
        return self.resS1Database[strat]

    def getResDict(self):
        return self.resS1Database

    def getBestStrat(self):
        return self.topStrategies[0]

    def getBest3Strats(self):
        return self.topStrategies
