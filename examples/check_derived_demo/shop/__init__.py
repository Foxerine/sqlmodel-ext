"""
Demo package for ``python -m sqlmodel_ext.check_derived`` (experimental).

It contains two deliberate mistakes that plain basedpyright does not report,
because ``partial=True`` changes the field types only at runtime. See
``examples/README.md`` for the command and its output.
"""
