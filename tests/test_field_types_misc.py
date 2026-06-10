"""
Misc field-type tests: ``IPAddress``, ``DirectoryPathType`` / ``FilePathType``,
and ``ModuleNameMixin``.

Covers:

1. ``IPAddress``: IPv4/IPv6 validation, invalid-format rejection, the value
   stays a plain ``str``, JSON serialization, and the ``is_private()`` helper
   on direct ``IPAddress(...)`` instances.
2. ``DirectoryPathType`` / ``FilePathType`` (via ``tmp_path``): validated
   values behave as ``pathlib.Path``; a file path must carry a filename
   component, a directory path must not carry an extension; string
   serialization; the internal ``_PathAsSQLString`` TypeDecorator round-trip
   and its use as the SA column type on a table model.
3. ``ModuleNameMixin``: auto-injects the caller's module ``__name__`` into the
   configured field when the kwarg is omitted; explicit kwargs win; the
   ``_module_name_field`` override is honored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from sqlmodel_ext import SQLModelBase, TableBaseMixin
from sqlmodel_ext.field_types import DirectoryPathType, FilePathType, IPAddress
from sqlmodel_ext.field_types._internal.path import _PathAsSQLString
from sqlmodel_ext.field_types.mixins import ModuleNameMixin


class FtIpModel(SQLModelBase):
    ip: IPAddress


class FtPathModel(SQLModelBase):
    directory: DirectoryPathType = Path(".")
    file: FilePathType = Path("placeholder.txt")


class FtPathColumnTable(SQLModelBase, TableBaseMixin, table=True):
    fpath: FilePathType = Path("a.txt")
    dpath: DirectoryPathType = Path("a")


class FtModuleNamed(ModuleNameMixin, SQLModelBase):
    name: str = ""


class FtModuleNamedCustomField(ModuleNameMixin, SQLModelBase):
    _module_name_field: ClassVar[str] = "source_module"
    source_module: str = ""
    name: str = "unrelated"


# ============================================================
# 1. IPAddress
# ============================================================

@pytest.mark.parametrize("ip", [
    "192.168.1.1",
    "8.8.8.8",
    "0.0.0.0",
    "255.255.255.255",
    "::1",
    "2001:db8::1",
    "fe80::1",
])
def test_ip_address_accepts_valid(ip: str) -> None:
    m = FtIpModel(ip=ip)
    assert m.ip == ip
    assert isinstance(m.ip, str)


@pytest.mark.parametrize("bad", [
    "999.1.1.1",
    "1.2.3",
    "1.2.3.4.5",
    "not-an-ip",
    "",
    "192.168.1.1/24",   # CIDR is not a bare address
    "2001:db8::zzzz",
])
def test_ip_address_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValidationError):
        FtIpModel(ip=bad)


def test_ip_address_json_serialization() -> None:
    m = FtIpModel(ip="10.20.30.40")
    assert json.loads(m.model_dump_json()) == {"ip": "10.20.30.40"}


def test_ip_address_is_private_helper() -> None:
    assert IPAddress("10.0.0.1").is_private() is True
    assert IPAddress("192.168.0.5").is_private() is True
    assert IPAddress("127.0.0.1").is_private() is True
    assert IPAddress("8.8.8.8").is_private() is False
    assert IPAddress("::1").is_private() is True
    assert IPAddress("2606:4700:4700::1111").is_private() is False


# ============================================================
# 2. DirectoryPathType / FilePathType
# ============================================================

class TestPathTypes:
    def test_file_path_accepts_real_file_path(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        m = FtPathModel(file=target)
        assert isinstance(m.file, Path)
        assert m.file == target

    def test_file_path_accepts_string_input(self) -> None:
        m = FtPathModel(file="some/dir/file.txt")
        assert isinstance(m.file, Path)
        assert m.file.name == "file.txt"

    def test_file_path_accepts_extensionless_filename(self) -> None:
        # Only a filename *component* is required, not an extension
        m = FtPathModel(file="some/dir/Makefile")
        assert m.file.name == "Makefile"

    @pytest.mark.parametrize("bad", [".", "..", ""])
    def test_file_path_rejects_paths_without_filename(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="filename"):
            FtPathModel(file=bad)

    def test_directory_path_accepts_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "subdir"
        m = FtPathModel(directory=target)
        assert isinstance(m.directory, Path)
        assert m.directory == target

    def test_directory_path_rejects_file_extension(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="extension"):
            FtPathModel(directory=tmp_path / "report.pdf")

    def test_directory_path_accepts_string_input(self) -> None:
        m = FtPathModel(directory="var/log/app")
        assert m.directory == Path("var/log/app")

    def test_path_serializes_to_string(self, tmp_path: Path) -> None:
        f = tmp_path / "out.txt"
        d = tmp_path / "logs"
        m = FtPathModel(file=f, directory=d)
        parsed = json.loads(m.model_dump_json())
        assert parsed["file"] == str(f)
        assert parsed["directory"] == str(d)

    def test_path_as_sql_string_bind_and_result_roundtrip(self) -> None:
        deco = _PathAsSQLString()
        assert deco.process_bind_param(Path("a/b/c.txt"), None) == str(Path("a/b/c.txt"))
        assert deco.process_bind_param(None, None) is None
        restored = deco.process_result_value("a/b/c.txt", None)
        assert restored == Path("a/b/c.txt")
        assert isinstance(restored, Path)
        assert deco.process_result_value(None, None) is None

    def test_path_columns_use_path_typedecorator(self) -> None:
        for col in ("fpath", "dpath"):
            col_type = FtPathColumnTable.__table__.c[col].type
            assert isinstance(col_type, _PathAsSQLString)


# ============================================================
# 3. ModuleNameMixin
# ============================================================

class TestModuleNameMixin:
    def test_auto_injects_caller_module_name(self) -> None:
        inst = FtModuleNamed()
        assert inst.name == __name__  # this test module

    def test_explicit_kwarg_wins(self) -> None:
        inst = FtModuleNamed(name="explicit")
        assert inst.name == "explicit"

    def test_custom_module_name_field(self) -> None:
        inst = FtModuleNamedCustomField()
        assert inst.source_module == __name__
        # the default 'name' field is untouched
        assert inst.name == "unrelated"

    def test_custom_field_explicit_value_wins(self) -> None:
        inst = FtModuleNamedCustomField(source_module="manual")
        assert inst.source_module == "manual"
