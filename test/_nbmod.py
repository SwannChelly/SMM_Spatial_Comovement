"""Load the reporting modules into a gate's namespace, with the gate's fixture stubs
installed where the library functions will resolve them.

Before the refactor each gate assembled the notebook's definition cells by content
anchor and `exec`'d them into its own globals, so a stub defined above the `exec` was
the one the library functions saw. The modules keep that semantic exactly, and more
cheaply: a function resolves a global from ITS OWN module, so a stub is installed by
setting the module attribute. Copying the library names back into the gate under a
`k not in g` guard then leaves the fixture in place, which is what the old `exec` did
for every name the executed cells did not themselves define.

This also retires three text-slicing hacks the cell layout forced on the gates --
pulling a function body out of a cell by string index, because executing the whole cell
would have overwritten the fixture's stubs. A module has no such problem: import the
function, stub the module attribute it reads.
"""
import importlib


def install(g, modules, stubs="auto"):
    """Import `modules`, install `stubs` on each, and copy their names into `g`.

    `g` is the calling gate's `globals()`. Names already bound there -- the fixture --
    are never overwritten.

    `stubs="auto"` installs every name the gate has ALREADY bound that the module also
    defines. That is precisely the old `exec(code, globals())` semantic: whatever the
    fixture set up before the executed cells ran was what those cells saw. Listing the
    stubs by hand instead is supported and is what a gate should do when it wants to be
    explicit about which library global it is replacing.
    """
    mods = [importlib.import_module(m) if isinstance(m, str) else m for m in modules]
    for m in mods:
        pairs = ({k: v for k, v in g.items()
                  if not k.startswith("__") and hasattr(m, k)}
                 if stubs == "auto" else (stubs or {}))
        for k, v in pairs.items():
            setattr(m, k, v)
    for m in mods:
        for k in vars(m):
            if not k.startswith("__") and k not in g:
                g[k] = getattr(m, k)
    return mods
