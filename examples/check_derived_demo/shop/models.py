"""Models: a base DTO with a validator, and a PATCH DTO derived with ``partial=True``."""
from typing import Self

from pydantic import model_validator

from sqlmodel_ext import SQLModelBase, Str64, Str256


class ArticleBase(SQLModelBase):
    title: Str64
    subtitle: Str256 | None = None

    @model_validator(mode='after')
    def _normalize_subtitle(self) -> Self:
        # Correct here: in ArticleBase, subtitle is `Str256 | None`.
        # Mistake 1: ArticleUpdate inherits this validator, and there subtitle may
        # also be Unset -- `Unset is not None` is True, so `.strip()` runs on Unset.
        if self.subtitle is not None:
            self.subtitle = self.subtitle.strip()
        return self

    def headline(self) -> str:
        # Fine on ArticleBase; wrong on ArticleUpdate (title may be Unset), but a
        # PATCH DTO never calls it, so check_derived reports it as latent only.
        return self.title.upper()


class ArticleUpdate(ArticleBase, partial=True):
    """PATCH body: every field of ArticleBase, each one omissible."""
