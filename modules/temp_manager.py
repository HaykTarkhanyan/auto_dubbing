import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager


class TempManager:
    def __init__(self, base_dir: str = ""):
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
        self.work_dir: Path | None = None

    def create_session(self) -> Path:
        self.work_dir = Path(tempfile.mkdtemp(prefix="autodub_", dir=self.base_dir))
        return self.work_dir

    def get_path(self, filename: str) -> str:
        if not self.work_dir:
            raise RuntimeError("No active session. Call create_session() first.")
        return str(self.work_dir / filename)

    def subdirectory(self, name: str) -> Path:
        if not self.work_dir:
            raise RuntimeError("No active session. Call create_session() first.")
        sub = self.work_dir / name
        sub.mkdir(exist_ok=True)
        return sub

    def cleanup(self):
        if self.work_dir and self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir = None

    @contextmanager
    def session(self):
        try:
            self.create_session()
            yield self
        finally:
            self.cleanup()
