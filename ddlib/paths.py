"""Find (or fetch) the repository so that notebooks work locally and on Google Colab without editing paths.

get_repo() tries three things, in this order:
  1. Already inside the repo: walk up from the current folder (and $DD_REPO if set) looking for ddlib/ + data/.
  2. git clone of the GitHub repo. If the repo is private, a GitHub token is read from the Colab secret GH_TOKEN
     or the environment variable GH_TOKEN. The token is only used for the clone and is removed from .git/config.
  3. Zip upload: on Colab, asks you to upload a zip of the repo (GitHub: Code > Download ZIP) and unpacks it.
Nothing is hard-coded except the public GitHub address and the Colab working folder /content.
"""
import os
import pathlib
import subprocess
import sys
import zipfile

REPO_URL = "https://github.com/SheridanLu/detecting-the-detector"
REPO_NAME = "detecting-the-detector"


def _is_repo(p):
    p = pathlib.Path(p)
    return (p / "ddlib" / "__init__.py").is_file() and (p / "data").is_dir()


def _find_upwards(start):
    p = pathlib.Path(start).resolve()
    for q in [p, *p.parents]:
        if _is_repo(q):
            return q
    return None


def _in_colab():
    return ("google.colab" in sys.modules) or ("COLAB_RELEASE_TAG" in os.environ)


def _read_token():
    tok = None
    try:
        from google.colab import userdata  # only exists on Colab
        try:
            tok = userdata.get("GH_TOKEN")
        except Exception:
            tok = None   # secret missing or notebook not granted access
    except ImportError:
        pass
    return tok or os.environ.get("GH_TOKEN") or None


def _clone(dest, url=REPO_URL):
    tok = _read_token()
    auth_url = url.replace("https://", f"https://x-access-token:{tok}@") if tok else url
    r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", auth_url, str(dest)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                       env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    if r.returncode != 0:
        msg = r.stderr.replace(tok, "***") if tok else r.stderr
        print(f"git clone failed ({'with' if tok else 'without'} token): {msg.strip()[:300]}")
        return False
    if tok:   # do not leave the token in .git/config
        subprocess.run(["git", "-C", str(dest), "remote", "set-url", "origin", url], check=False)
    return True


def _zip_fallback(base):
    instructions = (
        "Could not find or clone the repository.\n"
        "Option A: make the GitHub repo readable (public), or add a Colab secret named GH_TOKEN containing a GitHub\n"
        "          token with read access (key icon in the left sidebar), switch on 'Notebook access', then re-run.\n"
        f"Option B: on GitHub open {REPO_URL}, click Code > Download ZIP, then upload that zip when asked below.\n"
        "Option C (local Jupyter): start Jupyter from inside the repo folder, or set the environment variable\n"
        "          DD_REPO=/path/to/repo before starting it.")
    print(instructions)
    if not _in_colab():
        raise FileNotFoundError("Repository not found. See the instructions printed above.")
    from google.colab import files  # noqa
    up = files.upload()
    zips = [k for k in up if k.endswith(".zip")]
    if not zips:
        raise FileNotFoundError("No .zip uploaded.")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(base)
    found = [p.parent.parent for p in pathlib.Path(base).glob("*/ddlib/__init__.py")]
    found = [p for p in found if _is_repo(p)]
    if not found:
        raise FileNotFoundError("The zip does not contain ddlib/ and data/ folders.")
    return found[0]


_ROOT = None


def get_repo(clone_to=None, verbose=True, allow_clone=True):
    """Return the repository root as a pathlib.Path and put it on sys.path (so `import ddlib` works)."""
    global _ROOT
    if _ROOT is not None and _is_repo(_ROOT):
        return _ROOT
    root = None
    env = os.environ.get("DD_REPO")
    if env and _is_repo(env):
        root = pathlib.Path(env).resolve()
    if root is None:
        root = _find_upwards(os.getcwd())
    if root is None:
        here = _find_upwards(pathlib.Path(__file__).parent)   # ddlib imported from inside a repo checkout
        root = here
    if root is None:
        if not allow_clone:
            raise FileNotFoundError("Repository not found (run the first notebook cell, or set DD_REPO).")
        base = pathlib.Path(clone_to) if clone_to else (pathlib.Path("/content") if _in_colab() else pathlib.Path.cwd())
        dest = base / REPO_NAME
        if _is_repo(dest):
            root = dest
        elif not dest.exists() and _clone(dest) and _is_repo(dest):
            root = dest
        else:
            if dest.exists() and not _is_repo(dest):
                print(f"'{REPO_NAME}' exists but has no ddlib/ + data/ folders (old version of the repo?).")
            root = _zip_fallback(base)
    root = pathlib.Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _ROOT = root
    if verbose:
        print("Repository found. Data folder present:", (root / "data").is_dir())
    return root


def data_dir(root=None):
    """Path of data/. Never clones: uses the root found by get_repo() or the checkout this package lives in."""
    return pathlib.Path(root or get_repo(verbose=False, allow_clone=False)) / "data"


def rel(path):
    """Path relative to the repo root, for printing (keeps personal folder names out of notebook outputs)."""
    try:
        return str(pathlib.Path(path).resolve().relative_to(get_repo(verbose=False, allow_clone=False)))
    except ValueError:
        return pathlib.Path(path).name
