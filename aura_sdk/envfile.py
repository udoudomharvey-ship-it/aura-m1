"""Reads the user's anthropic.env (the same file the bash launcher sources) so Windows and Python entry
points get the key and voice settings too. Only sets variables that are not already set. Never prints values."""
import os
from . import platform_compat as compat


def _clean(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    home = os.path.expanduser("~")
    return os.path.expandvars(v.replace("$HOME", home).replace("${HOME}", home))


def parse(text: str) -> dict:
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        elif line.lower().startswith("set "):
            line = line[4:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.replace("_", "").isalnum() and k.upper() == k:
            out[k] = _clean(v)
    return out


def load_env_file(path: str = None, env=None) -> list:
    """Returns the names of the variables it set."""
    env = os.environ if env is None else env
    path = path or os.path.join(compat.config_dir(), "anthropic.env")
    try:
        with open(path, encoding="utf-8") as f:
            vals = parse(f.read())
    except OSError:
        return []
    set_now = []
    for k, v in vals.items():
        if v and not env.get(k):
            env[k] = v; set_now.append(k)
    return set_now
