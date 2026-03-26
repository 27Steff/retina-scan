"""
conftest.py
-----------
Agrega el directorio raíz del proyecto al path de Python para que
los tests puedan importar los módulos fuente directamente.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
