"""Module name auto-injection mixin."""
import inspect
from typing import ClassVar


class ModuleNameMixin:
    """
    A Mixin that automatically sets a field to the caller's module __name__
    when the model is instantiated.

    Usage:
    1. Inherit this Mixin in your model.
    2. Define a string field (e.g. ``name: str``).
    3. (Optional) Set ``_module_name_field`` if your field is not named 'name'.
    """
    _module_name_field: ClassVar[str] = "name"

    def __init__(self, **kwargs):
        field_to_set = self._module_name_field

        if field_to_set not in kwargs:
            try:
                caller_frame_record = inspect.stack()[1]
                module = inspect.getmodule(caller_frame_record.frame)

                if module and hasattr(module, '__name__'):
                    kwargs[field_to_set] = module.__name__
                else:
                    kwargs[field_to_set] = '__main__'
            except (IndexError, AttributeError):
                kwargs[field_to_set] = '__unknown__'

        super().__init__(**kwargs)
