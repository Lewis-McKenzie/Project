from typing import List, Set, Dict
import numpy as np

from models import BasicModel

POSITIVE = 0
NEUTRAL = 1
NEGATIVE = 2

class Argument:
    def __init__(self, model_results: Dict[str, Dict[str, List[float]]], alpha: float) -> None:
        self.model_results = model_results
        self.alpha = alpha

    def args_that_attack(self, argument: str, category: str, polarity: int) -> List[str]:
        attackers: List[str] = []
        opposite_polarity = 2 - polarity
        for text in self.arguments_with_category(category):
            if text == argument:
                return attackers
            if self.model_results[text][category][opposite_polarity] >= self.alpha:
                attackers.append(text)
            else:
                attackers = []
        return attackers


    def attack(self, argument: str) -> List[str]:
        attackers = []
        if not argument in self.model_results.keys():
            return []         
        arguments = self.model_results[argument]
        for category in arguments.keys():
            for arg in self.arguments_with_category(category):
                if self.does_attack(arg, argument, category):
                    attackers.append(arg)
        return attackers

    def arguments_with_category(self, category: str) -> List[str]:
        args = []
        for text, opinions in self.model_results.items():
            if category in opinions.keys():
                args.append(text)
        return args

    def does_attack(self, attacker: str, argument: str, category: str) -> bool:
        if not category in self.model_results[argument].keys() or not category in self.model_results[attacker].keys():
            return False
        pol1 = self.model_results[argument][category]
        pol2 = self.model_results[attacker][category]
        return (self.is_positive(pol1) and self.is_negative(pol2)) or (self.is_positive(pol2) and self.is_negative(pol1))

    def is_positive(self, polarity: List[float]) -> bool:
        return polarity[POSITIVE] >= self.alpha

    def is_negative(self, polarity: List[float]) -> bool:
        return polarity[NEGATIVE] >= self.alpha

    #TODO: algorithm only works on digraphs
    def fuzzy_labeling(self, iters: int) -> Dict[str, Dict[str, List[float]]]:
        fuzzy_labels = self.init_fuzzy_labels()
        for _ in range(iters):
            new_labels = {}
            for text, arguments in fuzzy_labels.items():
                arg = {}
                for category in arguments.keys():
                    arg[category] = [self.update_acceptability(fuzzy_labels, text, category, p) for p in range(3)]
                new_labels[text] = arg
            fuzzy_labels = new_labels.copy()
        return fuzzy_labels

    def init_fuzzy_labels(self) -> Dict[str, Dict[str, List[float]]]:
        fuzzy_labels = dict()
        for text, arguments in self.model_results.items():
            arg = {}
            for category in arguments.keys():
                arg[category] = [self.trustworthy_degree(text, category, p) for p in range(3)]
            fuzzy_labels[text] = arg
        return fuzzy_labels

    def trustworthy_degree(self, argument: str, category: str, polarity: int) -> float:
        return 1.0 * self.model_results[argument][category][polarity]
    
    def update_acceptability(self, fuzzy_labels: Dict[str, Dict[str, List[float]]], argument: str, category: str, polarity: int) -> float:
        if polarity == NEUTRAL:
            return fuzzy_labels[argument][category][polarity]
        if category == "AMBIENCE#GENERAL":
            pass
        args = self.arguments_with_category(category)
        opposite_polarity = 2 - polarity
        max_pol = 0.0
        for arg in self.args_that_attack(argument, category, polarity):
            max_pol = max(max_pol, fuzzy_labels[arg][category][opposite_polarity])
        return min(self.trustworthy_degree(argument, category, polarity), (fuzzy_labels[argument][category][polarity] + 1 - max_pol)/2)
 
    def describe(self, fl: Dict[str, Dict[str, List[float]]]) -> None:
        for text, arguments in fl.items():
            print(text)
            for category in arguments.keys():
                print(category)
                print("\tPOSITIVE {} -> {}".format(self.model_results[text][category][POSITIVE], fl[text][category][POSITIVE]))
                print("\tNEGATIVE {} -> {}".format(self.model_results[text][category][NEGATIVE], fl[text][category][NEGATIVE]))
            print()
        print("-------------------")

    def describe_category(self, fl: Dict[str, Dict[str, List[float]]], category: str) -> None:
        for text, arguments in fl.items():
            if category in arguments.keys():
                print(text)
                print(category)
                print("\tPOSITIVE {} -> {}".format(self.model_results[text][category][POSITIVE], fl[text][category][POSITIVE]))
                print("\t\t{}".format(self.args_that_attack(text, category, POSITIVE)))
                print("\tNEGATIVE {} -> {}".format(self.model_results[text][category][NEGATIVE], fl[text][category][NEGATIVE]))
                print("\t\t{}".format(self.args_that_attack(text, category, NEGATIVE)))

                print()
        print("-------------------")

    def support(self, text: str) -> List[str]:
        return []

    