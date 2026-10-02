"""Run a command as its own macOS privacy (TCC) identity, so camera and microphone work over SSH.

    python -m assistant_robot_reachy.own_permissions -- <command> [args...]

macOS asks for camera and microphone permission on behalf of the *responsible* process: for a
program started from Terminal that is Terminal, for one started over SSH it is sshd, which can
never be granted access, so the camera fails ("permission has been denied") and the microphone
records silence. This wrapper starts the command with the responsibility disclaimed (the
`posix_spawn` attribute that macOS app launchers use), so the command (here: Python) is
responsible for itself. macOS then asks once, on the Mac's screen, whether Python may use the
camera and the microphone; after "Allow" it works from SSH too.

The command stays in this process group (group signals reach it); SIGTERM and SIGINT sent to
the wrapper are passed on, and the wrapper exits with the command's exit code. On other
systems the command is simply exec'd.
"""

import contextlib
import ctypes
import os
import signal
import sys

_ENV_FLAG = "ASSISTANT_OWN_PERMISSIONS"


def _spawn_disclaimed(argv: list[str]) -> int:
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    attr = ctypes.c_void_p()
    if libc.posix_spawnattr_init(ctypes.byref(attr)) != 0:
        raise OSError(ctypes.get_errno(), "posix_spawnattr_init failed")
    if libc.responsibility_spawnattrs_setdisclaim(ctypes.byref(attr), 1) != 0:
        raise OSError(ctypes.get_errno(), "responsibility_spawnattrs_setdisclaim failed")
    c_argv = (ctypes.c_char_p * (len(argv) + 1))(*[a.encode() for a in argv], None)
    env = [f"{k}={v}".encode() for k, v in os.environ.items()]
    c_env = (ctypes.c_char_p * (len(env) + 1))(*env, None)
    pid = ctypes.c_int()
    path = argv[0].encode()
    rc = libc.posix_spawnp(ctypes.byref(pid), path, None, ctypes.byref(attr), c_argv, c_env)
    if rc != 0:
        raise OSError(rc, f"posix_spawn {argv[0]} failed: {os.strerror(rc)}")
    return pid.value


def run(argv: list[str]) -> int:
    """Run `argv` as its own TCC identity (macOS) and return its exit code."""
    if sys.platform != "darwin" or os.environ.get(_ENV_FLAG):
        os.execvp(argv[0], argv)
    os.environ[_ENV_FLAG] = "1"
    child = _spawn_disclaimed(argv)

    def forward(signum: int, frame: object) -> None:
        del frame
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)
    while True:
        try:
            _, status = os.waitpid(child, 0)
        except InterruptedError:
            continue
        return os.waitstatus_to_exitcode(status)


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    code = run(args)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
