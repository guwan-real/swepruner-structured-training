from .schemas import Relation, RelationType


def call_relations(relations: list[Relation]) -> list[Relation]:
    return [item for item in relations if item.relation in {RelationType.CALL, RelationType.CALLED_BY}]

