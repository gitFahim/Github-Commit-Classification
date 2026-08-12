"""Code change feature extraction.

Represents the 48 fine-grained AST-level change types used by this dataset.
These features are typically extracted by static analysis tools (e.g., ChangeDistiller,
GumTree) and provided as binary indicators in the dataset.
"""

from typing import List


class CodeChangeExtractor:
    """Placeholder for code change feature extraction.

    In the original paper, these features are extracted using AST differencing
    tools. This class documents the 48 change types used by this project and provides utilities
    for working with them.
    """

    CHANGE_TYPES = [
        "ADDING_ATTRIBUTE_MODIFIABILITY",
        "ADDING_CLASS_DERIVABILITY",
        "ADDING_METHOD_OVERRIDABILITY",
        "ADDITIONAL_CLASS",
        "ADDITIONAL_FUNCTIONALITY",
        "ADDITIONAL_OBJECT_STATE",
        "ALTERNATIVE_PART_DELETE",
        "ALTERNATIVE_PART_INSERT",
        "ATTRIBUTE_RENAMING",
        "ATTRIBUTE_TYPE_CHANGE",
        "CLASS_RENAMING",
        "COMMENT_DELETE",
        "COMMENT_INSERT",
        "COMMENT_MOVE",
        "COMMENT_UPDATE",
        "CONDITION_EXPRESSION_CHANGE",
        "DECREASING_ACCESSIBILITY_CHANGE",
        "DOC_DELETE",
        "DOC_INSERT",
        "DOC_UPDATE",
        "INCREASING_ACCESSIBILITY_CHANGE",
        "METHOD_RENAMING",
        "PARAMETER_DELETE",
        "PARAMETER_INSERT",
        "PARAMETER_ORDERING_CHANGE",
        "PARAMETER_RENAMING",
        "PARAMETER_TYPE_CHANGE",
        "PARENT_CLASS_CHANGE",
        "PARENT_CLASS_DELETE",
        "PARENT_CLASS_INSERT",
        "PARENT_INTERFACE_CHANGE",
        "PARENT_INTERFACE_DELETE",
        "PARENT_INTERFACE_INSERT",
        "REMOVED_CLASS",
        "REMOVED_FUNCTIONALITY",
        "REMOVED_OBJECT_STATE",
        "REMOVING_ATTRIBUTE_MODIFIABILITY",
        "REMOVING_CLASS_DERIVABILITY",
        "REMOVING_METHOD_OVERRIDABILITY",
        "RETURN_TYPE_CHANGE",
        "RETURN_TYPE_DELETE",
        "RETURN_TYPE_INSERT",
        "STATEMENT_DELETE",
        "STATEMENT_INSERT",
        "STATEMENT_ORDERING_CHANGE",
        "STATEMENT_PARENT_CHANGE",
        "STATEMENT_UPDATE",
        "UNCLASSIFIED_CHANGE",
    ]

    @classmethod
    def get_change_types(cls) -> List[str]:
        """Return the list of 48 AST-level change types."""
        return cls.CHANGE_TYPES.copy()

    @classmethod
    def count(cls) -> int:
        """Return the number of change types."""
        return len(cls.CHANGE_TYPES)
