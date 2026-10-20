import numpy as np


def flatten_prototype_indices(
    prototype_indices_by_class,
) -> list[int]:
    """
    Flatten {class: [indices]} into a single list.
    """
    indices = []

    for class_indices in prototype_indices_by_class.values():
        indices.extend(class_indices)

    return [int(i) for i in indices]
