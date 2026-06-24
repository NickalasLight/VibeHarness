"""Tool-call codecs.

Each module here is one isolated wire format named ``<name>_codec`` and exposes a
module-level ``CODEC`` instance; :func:`vibeharness.codec.get_codec` imports them by
name. Keeping every format in its own file lets parallel work add codecs without
sharing a code file (and so without merge conflicts).
"""
