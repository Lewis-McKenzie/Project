from typing import List, Set

from models import BasicModel

class Argument:
    def __init__(self, model: BasicModel, data: List[str], labels: List[List[float]], alpha: float) -> None:
        self.model = model
        self.data = data
        self.labels = labels
        self.category_labels = self.model.invert_all(self.labels, alpha)
    
    def change_alpha(self, alpha: float) -> None:
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

    def support(self, text: str) -> List[str]:
        return []

    