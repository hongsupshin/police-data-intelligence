from unittest.mock import patch

import pytest

from src.run import main, run


@patch("sys.argv", ["src.run", "12345", "civilians_shot"])
@patch("src.run.run")
def test_arg_parsing(mock_run) -> None:

    mock_run.return_value = {"outcome_summary": "test_summary"}
    main()
    mock_run.assert_called_once_with("12345", "civilians_shot")


@patch("sys.argv", ["src.run", "12345", "invalid_dataset_type"])
def test_invalid_dataset_type() -> None:

    with pytest.raises(ValueError, match="Invalid DatasetType: 'invalid_dataset_type'"):
        main()


@patch("src.run.os.environ", {"ANTHROPIC_API_KEY": "fake-key"})
@patch("src.run.Settings")
@patch("src.run.build_graph")
@patch("src.run.SqliteSaver")
@patch("src.run.sqlite3")
@patch("src.run.ChatAnthropic")
def test_run(mock_chat, mock_sqlite3, mock_saver, mock_build, mock_settings) -> None:

    mock_build.return_value.invoke.return_value = {"outcome_summary": "done"}
    assert run("12345", "civilians_shot")["outcome_summary"] == "done"
