from typing import List, Set, Dict
import numpy as np

from models import BasicModel

POSITIVE = 0
NEGATIVE = 2

class Argument:
    def __init__(self, model_results: Dict[str, Dict[str, List[float]]], alpha: float) -> None:
        self.model_results = model_results
        self.alpha = alpha

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
    def fuzzy_labeling(self, iters: int) -> List[List[float]]:
        fuzzy_labels = self.init_fuzz_labels()
        new_labels = fuzzy_labels.copy()
        for _ in range(iters):
            for i, _ in enumerate(self.data):
                new_labels[i] = self.acceptability_step(i, fuzzy_labels)
            fuzzy_labels = new_labels.copy()
        self.describe(fuzzy_labels)
        return fuzzy_labels

    def init_fuzz_labels(self) -> List[List[float]]:
        labels = []
        for label in self.labels:
            l = [i if i >= self.alpha else 0.0 for i in label]
            labels.append(l)
        return labels

    def acceptability_step(self, target_index: int, fuzzy_labels: List[List[float]]) -> List[float]:
        hot_indices: List[int] = np.argwhere(self.labels[target_index] >= self.alpha)[..., 0]
        # remove neutrals
        hot_indices = [i for i in hot_indices if i != 0 and (i < 13 or i > 24)]
        fuzzy_label = fuzzy_labels[target_index].copy()
        for hot_index in hot_indices:
            actual_label_val = self.labels[target_index][hot_index]            
            fuzzy_label[hot_index] = min(actual_label_val, self.new_label(target_index, hot_index, fuzzy_labels))
        return fuzzy_label

    def new_label(self, target_index: int, hot_index: int, fuzzy_labels: List[List[float]]) -> float:
        fuzzy_label_val = fuzzy_labels[target_index][hot_index]
        max_attack = self.max_attack(target_index, hot_index, fuzzy_labels)
        max_support = self.max_support(hot_index, fuzzy_labels)
        avg_attack = self.avg_attack(hot_index, fuzzy_labels)
        sum_attack = self.sum_attack(hot_index, fuzzy_labels)
        total_attack = self.total_attack(hot_index, fuzzy_labels)
        sum_support = self.sum_support(target_index, hot_index, fuzzy_labels)
        total_support = self.total_support(target_index, hot_index, fuzzy_labels)
        ratio_support = total_support/(total_support+total_attack)
        return (ratio_support * fuzzy_label_val + (1-ratio_support) * ((1 - max_attack)))
            
    def max_attack(self, target_index: int, category_index, fuzzy_labels: List[List[float]]) -> float:
        opposite_index = self.opposite_index(category_index)
        max_attack = 0.0
        for i, label in enumerate(fuzzy_labels):
            if i == target_index:
                continue
            max_attack = max(max_attack, label[opposite_index])
        return max_attack

    def max_support(self, category_index, fuzzy_labels: List[List[float]]) -> float:
        max_support = 0.0
        for _, label in enumerate(fuzzy_labels):
            max_support = max(max_support, label[category_index])
        return max_support

    def avg_attack(self, category_index, fuzzy_labels: List[List[float]]) -> float:
        s = 0.0
        t = 0
        opposite_index = self.opposite_index(category_index)
        for _, label in enumerate(fuzzy_labels):
            if label[opposite_index] != 0:
                t += 1
            s += label[opposite_index]
        t = max(t, 1)
        return s / t

    def sum_attack(self, category_index, fuzzy_labels: List[List[float]]) -> float:
        s = 0.0
        opposite_index = self.opposite_index(category_index)
        for _, label in enumerate(fuzzy_labels):
            s += label[opposite_index]
        return s

    def sum_support(self, target_index: int, category_index, fuzzy_labels: List[List[float]]) -> float:
        s = 0.0
        for i, label in enumerate(fuzzy_labels):
            s += label[category_index]
        return s

    def total_attack(self, category_index, fuzzy_labels: List[List[float]]) -> float:
        t = 0
        opposite_index = self.opposite_index(category_index)
        for _, label in enumerate(fuzzy_labels):
            if label[opposite_index] != 0:
                t += 1
        return t

    def total_support(self, target_index: int, category_index, fuzzy_labels: List[List[float]]) -> float:
        t = 0
        for i, label in enumerate(fuzzy_labels):
            if label[category_index] != 0:
                t += 1
        return t

    def opposite_index(self, index: int) -> int:
        if index == 0:
            return index
        elif index < 13:
            return 24 + index
        elif index > 24:
            return index - 24
        else:
            return index

    def describe(self, fl: List[List[float]]) -> None:
        labels = self.init_fuzz_labels()
        for i, l in enumerate(fl):
            for hot in np.argwhere(self.labels[i] >= self.alpha)[..., 0]:
                cat = np.take(self.model.encoder.get_vocabulary(), hot)
                print(f"{cat}: {labels[i][hot]} -> {l[hot]}")
            print()
        print("-------------------")

    def support(self, text: str) -> List[str]:
        return []

    