import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if os.path.isdir(os.path.join(REPO_ROOT, "src")):
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
