from crane_project.tools import patch_dinov2_py38 as patcher


def test_patch_source_adds_future_import_after_docstring():
    source = '"""module docs"""\n\nclass A:\n    value: float | None = None\n'
    updated, changed = patcher.patch_source(source)
    assert changed is True
    assert updated.startswith(
        '"""module docs"""\nfrom __future__ import annotations\n')


def test_patch_source_is_idempotent():
    source = ('from __future__ import annotations\n'
              'value: float | None = None\n')
    updated, changed = patcher.patch_source(source)
    assert changed is False
    assert updated == source


def test_patch_source_handles_python39_builtin_generics():
    source = 'def collect(values: list[str]) -> dict[str, int]:\n    return {}\n'
    updated, changed = patcher.patch_source(source)
    assert changed is True
    assert updated.startswith(patcher.FUTURE_IMPORT)


def test_patch_repo_changes_only_pep604_files(tmp_path):
    repo = tmp_path / 'dinov2'
    package = repo / 'dinov2'
    package.mkdir(parents=True)
    (repo / 'hubconf.py').write_text('dependencies = ["torch"]\n')
    target = package / 'attention.py'
    target.write_text('class A:\n    value: float | None = None\n')
    plain = package / 'plain.py'
    plain.write_text('value = 1\n')
    result = patcher.patch_repo(str(repo))
    assert result['changed_count'] == 1
    assert result['changed_files'] == ['dinov2/attention.py']
    assert patcher.FUTURE_IMPORT in target.read_text()
    assert patcher.FUTURE_IMPORT not in plain.read_text()
    second = patcher.patch_repo(str(repo))
    assert second['changed_count'] == 0
