# conftest.py — configuração raiz do pytest
# Garante que o diretório raiz está no sys.path para todos os testes.

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
