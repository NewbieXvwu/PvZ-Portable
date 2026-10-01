#!/usr/bin/env python3
"""Run a command on a remote Windows desktop over SSH without quoting hell.

The chain is ``ssh -> cmd.exe -> wsl -> bash`` and every layer re-parses quotes.
The payload is therefore delivered over **stdin**, never as command-line
arguments: cmd re-parses the argument string, and ``powershell -EncodedCommand
<base64>`` silently fails through it (exit 1, no output) because the trailing
``=`` padding and ``+/`` characters do not survive.  See ``_ssh`` for the full
account -- this bit the repository once already.

The target host is deliberately **not** hard-coded -- this repository is public
and a LAN address plus username is nobody else's business.  Supply it through
``--host`` or the ``PVZ_DESKTOP_HOST`` environment variable::

    export PVZ_DESKTOP_HOST=user@10.0.0.5
    python3 win_ssh.py ps  'Get-ChildItem C:\\'
    python3 win_ssh.py ps  --file probe.ps1
    python3 win_ssh.py wsl 'python3 -c "import torch; print(torch.__version__)"'
    python3 win_ssh.py wsl --file probe.py --interpreter python3
    python3 win_ssh.py wsl --file probe.py --venv ~/ml

If the desktop's sshd does not listen on 22 -- a forwarded or tunnelled port is the
common case -- pass ``--port`` or set ``PVZ_DESKTOP_PORT``.  Two non-ASCII details
matter: the Windows console code page is GBK on a Chinese install (mojibake on the
way back unless UTF-8 is forced), and ``&&`` is not a statement separator before
PowerShell 7.

Two more traps worth knowing before writing a remote script:

* The WSL wrapper runs under ``set -euo pipefail``, so **any** command that exits
  non-zero aborts the whole script.  A ``grep`` with no match will do it -- append
  ``|| true`` when a miss is expected.
* ``~`` is expanded by the *local* shell if it reaches a quoted argument, so
  ``--venv ~/.venvs/ml`` must be quoted (``--venv '~/.venvs/ml'``); ``_venv_source``
  rewrites it through ``$HOME`` on the far side.
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
from pathlib import Path

DEFAULT_DISTRO = "Ubuntu"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]

HOST = ""
PORT = ""
DISTRO = DEFAULT_DISTRO


def _ssh(argv: list[str], stdin: str | None = None) -> int:
    """Run *argv* on the remote host, optionally feeding *stdin* to it.

    The payload is delivered over **stdin**, never as command-line arguments.
    The remote login shell is ``cmd.exe``, not PowerShell: cmd re-parses the
    argument string, and ``powershell -EncodedCommand <base64>`` silently fails
    (exit 1, no output) through it -- the trailing ``=`` padding and ``+/``
    characters do not survive.  Piping the script in sidesteps cmd, PowerShell
    and ``wsl.exe`` quoting entirely, and also avoids the UTF-8 BOM that
    ``wsl.exe`` prepends to anything arriving on a pipe *from PowerShell*.
    """
    port_opts = ["-p", PORT] if PORT else []
    return subprocess.run(
        ["ssh", *port_opts, *SSH_OPTS, HOST, *argv],
        input=stdin.encode("utf-8") if stdin is not None else None,
        check=False,
    ).returncode


def run_ps(script: str) -> int:
    # ``$ProgressPreference`` keeps PowerShell from serialising progress records
    # into the stderr stream, which SSH renders as ``#< CLIXML`` noise.  The two
    # encoding lines stop a GBK console code page from turning any non-ASCII
    # output into mojibake on the way back.
    script = (
        "$ProgressPreference='SilentlyContinue'; "
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$OutputEncoding=[System.Text.Encoding]::UTF8; "
    ) + script
    return _ssh(["powershell", "-NoProfile", "-Command", "-"], script)


def _venv_source(venv: str) -> str:
    # ``~`` inside quotes is not expanded by bash, so go through $HOME.
    venv_path = venv.replace("~", '"$HOME"') if venv.startswith("~") else f'"{venv}"'
    return f"source {venv_path}/bin/activate"


def run_wsl(script: str, interpreter: str | None, venv: str | None) -> int:
    """Pipe *script* into ``bash`` inside WSL, optionally via a venv interpreter."""
    if interpreter:
        # Activate in the outer shell, then hand the decoded script to the
        # interpreter on its stdin so the script's own quoting stays untouched.
        outer = ["set -euo pipefail"]
        if venv:
            outer.append(_venv_source(venv))
        payload = base64.b64encode(script.encode("utf-8")).decode("ascii")
        outer.append(f"echo {payload} | base64 -d | {interpreter} -")
        bootstrap = "\n".join(outer) + "\n"
    else:
        inner = ["set -euo pipefail"]
        if venv:
            inner.append(_venv_source(venv))
        inner.append(script)
        payload = base64.b64encode("\n".join(inner).encode("utf-8")).decode("ascii")
        # base64 is alphanumeric plus ``+/=``, so it survives every layer.
        bootstrap = f"echo {payload} | base64 -d | bash\n"

    return _ssh(["wsl", "-d", DISTRO, "-e", "bash", "-s"], bootstrap)


def main() -> int:
    global HOST, PORT, DISTRO
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("ps", "wsl"))
    parser.add_argument("command", nargs="?", help="command text (or omit when using --file)")
    parser.add_argument("--file", type=Path, help="read the payload from this local file")
    parser.add_argument("--interpreter", help="wsl only: pipe the payload into this interpreter")
    parser.add_argument("--venv", help="wsl only: activate this venv directory first")
    parser.add_argument("--host", default=os.environ.get("PVZ_DESKTOP_HOST", ""),
                        help="user@address of the desktop (default: $PVZ_DESKTOP_HOST)")
    parser.add_argument("--port", default=os.environ.get("PVZ_DESKTOP_PORT", ""),
                        help="ssh port on the desktop (default: $PVZ_DESKTOP_PORT, else 22)")
    parser.add_argument("--distro", default=DEFAULT_DISTRO, help="WSL distribution name")
    args = parser.parse_args()

    HOST = args.host
    PORT = str(args.port)
    DISTRO = args.distro
    if not HOST:
        parser.error("no target host: pass --host user@address or set PVZ_DESKTOP_HOST")

    if args.file:
        script = args.file.read_text(encoding="utf-8")
    elif args.command:
        script = args.command
    else:
        parser.error("provide either a command or --file")

    if args.mode == "ps":
        return run_ps(script)
    return run_wsl(script, args.interpreter, args.venv)


if __name__ == "__main__":
    raise SystemExit(main())
