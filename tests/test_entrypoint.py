from pathlib import Path


def test_entrypoint_allows_long_polling_and_reasoning_turns():
    script = (Path(__file__).parents[1] / "entrypoint.sh").read_text()

    assert 'GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-90}"' in script
    assert '--timeout "$GUNICORN_TIMEOUT"' in script
