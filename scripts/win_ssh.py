#!/usr/bin/env python3
"""Run a command on a remote Windows desktop over SSH without quoting hell.

The chain is ``ssh -> PowerShell 5.1 -> wsl -> bash`` and every layer re-parses
quotes.  Instead of escaping, the payload is base64-encoded and handed to
PowerShell's ``-EncodedCommand`` (UTF-16LE) or piped into ``wsl ... bash``.

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
common case -- pass ``--port`` or set ``PVZ_DESKTOP_PORT``.  The login shell on the
Windows side is PowerShell, so the two non-ASCII details below matter: PowerShell 5.1
writes console output in the OEM code page (GBK on a Chinese install), which arrives
as mojibake unless the encoding is forced to UTF-8, and ``&&`` is not a statement
separator before PowerShell 7.
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


def _ssh(argv: list[str]) -> int:
    port_opts = ["-p", PORT] if PORT else []
    return subprocess.run(["ssh", *port_opts, *SSH_OPTS, HOST, *argv], check=False).returncode


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
    payload = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return _ssh(["powershell", "-NoProfile", "-OutputFormat", "Text", "-EncodedCommand", payload])


def run_wsl(script: str, interpreter: str | None, venv: str | None) -> int:
    """Pipe *script* into ``bash`` inside WSL, optionally via a venv interpreter."""
    preamble = ["set -euo pipefail"]
    if venv:
        # ``~`` inside quotes is not expanded by bash, so go through $HOME.
        venv_path = venv.replace("~", '"$HOME"') if venv.startswith("~") else f'"{venv}"'
        preamble.append(f'source {venv_path}/bin/activate')
    if interpreter:
        # Feed the script to the interpreter on stdin so its own quoting is untouched.
        preamble.append(f'exec {interpreter} -')

    wrapper = "\n".join(preamble) + "\n" + script
    payload = base64.b64encode(wrapper.encode("utf-8")).decode("ascii")
    # Piping a PowerShell string into ``wsl.exe`` prepends a UTF-8 BOM, and
    # ``base64 -d`` rejects that with "invalid input".  Passing the payload as an
    # argument to ``echo`` *inside* the WSL bash avoids the pipe altogether: base64
    # is alphanumeric plus ``+/=``, so it survives every quoting layer untouched.
    ps = f"wsl -d {DISTRO} -e bash -c 'echo {payload} | base64 -d | bash'"
    return run_ps(ps)


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
