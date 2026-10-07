from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run(**params):
    at = AppTest.from_file(APP, default_timeout=120)
    for k, v in params.items():
        at.query_params[k] = v
    return at.run()


def test_correct_ecg_passes_and_swap_is_caught():
    at = run(ecg="0")
    assert not at.exception
    assert at.success and "correctly placed" in at.success[0].value
    at = run(ecg="0", swap="LA/RA")
    assert at.error and "LA/RA" in at.error[0].value
