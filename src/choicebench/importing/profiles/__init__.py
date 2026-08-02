"""Paper-specific import profiles.

No generic importer module may import anything from this package, and no
paper-specific constant (cell IDs, method names, queue file names) may leak
into src/choicebench/importing/*.py outside this directory.
"""
