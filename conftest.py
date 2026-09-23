"""pytest 引导：确保仓库根目录在 sys.path 中。"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
