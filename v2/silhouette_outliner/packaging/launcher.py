"""PyInstaller entry point.

PyInstaller runs the entry script as ``__main__`` (no parent package), which
breaks the relative imports inside ``silhouette_outliner.gui.app``. This thin
launcher uses an absolute import so the package context is preserved.
"""

from silhouette_outliner.gui.app import main

if __name__ == "__main__":
    main()
