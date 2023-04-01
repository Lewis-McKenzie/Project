from typing import List, Set
import numpy as np

from models import BasicModel

class Argument:
    def __init__(self, model: BasicModel, data: List[str], labels: List[List[float]], alpha: float) -> None:
        self.model = model
        self.data = data
        self.labels = labels
        self.alpha = alpha
        self.category_labels = self.model.invert_all(self.labels, alpha)
    
    def change_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.category_labels = self.model.invert_all(self.labels, alpha)

    def attack(self, argument: str) -> List[str]:
        attackers = []
        index = self.data.index(argument)
        opposites = self.opposites(self.category_labels[index])
        for i, categories in enumerate(self.category_labels):
            if opposites.intersection(set(categories)) != set():
                attackers.append(self.data[i])
        return attackers

    def opposites(self, categories: List[str]) -> Set[str]:
        opposites: Set[str] = set()
        for category in categories:
            if "positive" in category:
                opposites.add(category.replace("positive", "negative"))
            elif "negative" in category:
                opposites.add(category.replace("negative", "positive"))
        return opposites

    def is_conflict_free(self, arguments: List[str]) -> bool:
        attacks: Set[str] = set() # set of categories attacked by the given arguments
        for text in arguments:
            index = self.data.index(text)
            # if the current item attacks any of the given arguments, the set is not conflict free
            if attacks.intersection(set(self.category_labels[index])) != set():
                return False
            opposites = self.opposites(self.category_labels[index])
            for category in opposites:
                attacks.add(category)
        return True

    def defends(self, arguments: List[str], target: str) -> bool:
        categories_defended = self.categories_of(arguments)
        target_categories = self.category_labels[self.data.index(target)]
        return categories_defended.intersection(target_categories) == set(target_categories)

    def defended_by(self, arguments: List[str]) -> Set[str]:
        categories_defended = self.categories_of(arguments)
        arguments_defended: Set[str] = set()
        for i, categories in enumerate(self.category_labels):
            if categories_defended.intersection(categories) == set(categories):
                arguments_defended.add(self.data[i])
        return arguments_defended

    def categories_of(self, arguments: List[str]) -> Set[str]:
        target_categories: Set[str] = set()
        for argument in arguments:
            for category in self.category_labels[self.data.index(argument)]:
                target_categories.add(category)
        return target_categories

    def is_admissable(self, arguments: List[str]) -> bool:
        return self.is_conflict_free(arguments) and set(arguments).issubset(self.defended_by(arguments))

    def is_complete_extension(self, arguments: List[str]) -> bool:
        return self.is_conflict_free(arguments) and set(arguments) == self.defended_by(arguments)

    def is_grounded_extension(self, arguments: List[str]) -> bool:
        return self.is_conflict_free(arguments)

    def is_preffered_extension(self, arguments: List[str]) -> bool:
        return self.is_conflict_free(arguments)

    def is_stable_etension(self, arguments: List[str]) -> bool:
        other_arguments = set(self.data).difference(arguments)
        categories_defended = self.categories_of(arguments)
        categories_attacked = self.opposites(list(categories_defended))
        for argument in other_arguments:
            categories = self.category_labels[self.data.index(argument)]
            if categories_attacked.intersection(categories) == set():
                return False
        return self.is_preffered_extension(arguments)
    

    def fuzzy_labeling(self, iters: int) -> List[List[float]]:
        fuzzy_labels = self.init_fuzz_labels()
        for _ in range(iters):
            for i, _ in enumerate(self.data):
                fuzzy_labels = self.acceptability_step(i, fuzzy_labels)
        return fuzzy_labels

    def init_fuzz_labels(self) -> List[List[float]]:
        labels = []
        for label in self.labels:
            l = [i if i >= self.alpha else 0.0 for i in label]
            labels.append(l)
        return labels

    def acceptability_step(self, target_index: int, fuzzy_labels: List[List[float]]) -> List[List[float]]:
        hot_indices: List[int] = np.argwhere(self.labels[target_index] >= self.alpha)[..., 0]
        # remove neutrals
        hot_indices = [i for i in hot_indices if i != 0 and (i < 13 or i > 24)]
        for hot_index in hot_indices:
            fuzzy_label = fuzzy_labels[target_index][hot_index]
            actual_label = self.labels[target_index][hot_index]
            max_attack = self.max_attack(target_index, hot_index, fuzzy_labels)
            fuzzy_labels[target_index][hot_index] = min(actual_label, (fuzzy_label + 1 - max_attack)/2)
        return fuzzy_labels
            
    def max_attack(self, target_index: int, category_index, fuzzy_labels: List[List[float]]) -> float:
        opposite_index = self.opposite_index(category_index)
        max_attack = 0.0
        for i, label in enumerate(fuzzy_labels):
            if i == target_index:
                continue
            max_attack = max(max_attack, label[opposite_index])
        return max_attack

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

    def support(self, text: str) -> List[str]:
        return []

    