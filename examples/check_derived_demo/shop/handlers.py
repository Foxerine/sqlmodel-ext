"""Consumers of the PATCH DTO: one wrong, one right."""
from shop.models import ArticleUpdate
from sqlmodel_ext import Unset


def subtitle_preview_wrong(patch: ArticleUpdate) -> str:
    # Mistake 2: `is not None` does not rule out "not provided" (Unset).
    if patch.subtitle is not None:
        return patch.subtitle[:20]
    return ''


def subtitle_preview_right(patch: ArticleUpdate) -> str:
    if patch.subtitle is Unset or patch.subtitle is None:
        return ''
    return patch.subtitle[:20]
