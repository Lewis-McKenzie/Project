from typing import List, Set

from models import BasicModel

class Argument:
    def __init__(self, model: BasicModel, data: List[str], labels: List[List[float]]) -> None:
        self.model = model
        self.data = data
        self.labels = labels

    def attack(self, argument: str, alpha: float) -> List[str]:
        attackers = []
        all_categories = self.model.invert_all(self.labels, alpha)
        index = self.data.index(argument)
        opposites = self.opposites(all_categories[index])
        for i, categories in enumerate(all_categories):
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

    def is_conflict_free(self, arguments: List[str], alpha: float) -> bool:
        attacks: Set[str] = set() # set of categories attacked by the given arguments
        all_categories = self.model.invert_all(self.labels, alpha)
        for text in arguments:
            index = self.data.index(text)
            # if the current item attacks any of the given arguments, the set is not conflict free
            if attacks.intersection(set(all_categories[index])) != set():
                return False
            opposites = self.opposites(all_categories[index])
            for category in opposites:
                attacks.add(category)
        return True

    def defends(self, arguments: List[str], target: str, alpha: float) -> bool:
        all_categories = self.model.invert_all(self.labels, alpha)
        categories_defended = self.categories_of(arguments, alpha)
        target_categories = all_categories[self.data.index(target)]
        return categories_defended.intersection(target_categories) == set(target_categories)

    def defended_by(self, arguments: List[str], alpha: float) -> Set[str]:
        categories_defended = self.categories_of(arguments, alpha)
        all_categories = self.model.invert_all(self.labels, alpha)
        arguments_defended: Set[str] = set()
        for i, categories in enumerate(all_categories):
            if categories_defended.intersection(categories) == set(categories):
                arguments_defended.add(self.data[i])
        return arguments_defended

    def categories_of(self, arguments: List[str], alpha: float) -> Set[str]:
        all_categories = self.model.invert_all(self.labels, alpha)
        target_categories: Set[str] = set()
        for argument in arguments:
            for category in all_categories[self.data.index(argument)]:
                target_categories.add(category)
        return target_categories

    def is_admissable(self, arguments: List[str], alpha: float) -> bool:
        return self.is_conflict_free(arguments, alpha) and set(arguments).issubset(self.defended_by(arguments, alpha))

    def is_complete_extension(self, arguments: List[str], alpha: float) -> bool:
        return self.is_conflict_free(arguments, alpha) and set(arguments) == self.defended_by(arguments, alpha)

    def is_grounded_extension(self, arguments: List[str], alpha: float) -> bool:
        return self.is_conflict_free(arguments, alpha)

    def is_preffered_extension(self, arguments: List[str], alpha: float) -> bool:
        return self.is_conflict_free(arguments, alpha)

    def is_stable_etension(self, arguments: List[str], alpha: float) -> bool:
        other_arguments = set(self.data).difference(arguments)
        categories_defended = self.categories_of(arguments, alpha)
        categories_attacked = self.opposites(list(categories_defended))
        all_categories = self.model.invert_all(self.labels, alpha)
        for argument in other_arguments:
            categories = all_categories[self.data.index(argument)]
            if categories_attacked.intersection(categories) == set():
                return False
        return self.is_preffered_extension(arguments, alpha)

    def support(self, text: str) -> List[str]:
        return []

    