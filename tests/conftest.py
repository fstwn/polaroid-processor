import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

# The scanner profiles file is a path that does not exist, from the start of the session, so that no test (not even a
# module-scoped fixture, which runs before the autouse fixture below) reads the user's own
# ``~/.config/polaroid-processor/profiles.yaml``. A test that needs profiles sets ``POLAROID_PROFILES`` itself.
_NO_PROFILES = str(Path(tempfile.gettempdir()) / "polaroid-tests-no-profiles" / "profiles.yaml")
os.environ["POLAROID_PROFILES"] = _NO_PROFILES


@pytest.fixture(autouse=True)
def _no_user_profiles(tmp_path, monkeypatch):
    """Per test: a non-existent profiles file inside the test's own temp folder."""
    monkeypatch.setenv("POLAROID_PROFILES", str(tmp_path / "no-profiles-here.yaml"))
