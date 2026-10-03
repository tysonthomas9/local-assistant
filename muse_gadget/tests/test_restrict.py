import socket

from musegadget.executor import Account

from gadget import restrict
from gadget.link import RobotService
from musegadget.identity import Identity


def test_registered_commands_exclude_shell_and_files():
    specs = restrict.command_specs()
    assert set(specs) == {"device.health"}
    for blocked in ("system.run", "file.read", "file.write"):
        assert blocked not in specs


def test_executor_refuses_blocked_commands(tmp_path):
    ex = restrict.RestrictedExecutor(Account.current())
    for command, params in (("system.run", {"command": "touch " + str(tmp_path / "x")}),
                            ("file.read", {"path": "/etc/hostname"}),
                            ("file.write", {"path": str(tmp_path / "y"), "data_b64": "eA==", "final": True})):
        result = ex.run(command, params)
        assert result == {"ok": False, "error": f"unsupported command: {command}"}
    assert list(tmp_path.iterdir()) == []


def test_health_hides_host_name(monkeypatch):
    monkeypatch.delenv(restrict.DISPLAY_NAME_ENV, raising=False)
    result = restrict.RestrictedExecutor(Account.current()).run("device.health", {})
    assert result["ok"]
    assert result["payload"]["hostname"] == "Reachy Mini"
    assert socket.gethostname() not in result["payload"].values()


def test_display_name_is_neutral(monkeypatch):
    monkeypatch.delenv(restrict.DISPLAY_NAME_ENV, raising=False)
    service = RobotService(identity=Identity("02:00:00:ab:cd:ef"),
                           executor=restrict.RestrictedExecutor(Account.current()))
    assert service.display_name == "Reachy Mini"
    assert service.display_name != socket.gethostname()
    monkeypatch.setenv(restrict.DISPLAY_NAME_ENV, "Desk Robot")
    assert restrict.display_name() == "Desk Robot"
    monkeypatch.setenv(restrict.DISPLAY_NAME_ENV, "   ")
    assert restrict.display_name() == "Reachy Mini"
