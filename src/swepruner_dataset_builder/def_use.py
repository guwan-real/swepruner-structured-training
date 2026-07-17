from .schemas import Relation, RelationType


def def_use_relations(relations: list[Relation]) -> list[Relation]:
    return [item for item in relations if item.relation == RelationType.DEF_USE]

