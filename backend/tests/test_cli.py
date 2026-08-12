from typer.testing import CliRunner

from panoptes.cli import app


def test_doctor() -> None:
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Panoptes environment" in result.output


def test_fixtures() -> None:
    result = CliRunner().invoke(app, ["fixtures"])
    assert result.exit_code == 0
    assert "human-prose" in result.output
