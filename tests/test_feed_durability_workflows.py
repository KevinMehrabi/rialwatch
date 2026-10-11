import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/intraday-collection.yml"


def step_script(name: str) -> str:
    step = WORKFLOW.read_text().split(f"      - name: {name}\n", 1)[1].split("      - name:", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


class FeedDurabilityWorkflowTests(unittest.TestCase):
    def test_distributed_and_manual_collections_do_not_wait_for_the_daily_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "args"
            for event, schedule, immediate in [
                ("schedule", "37 5,9,17,21 * * *", True),
                ("workflow_dispatch", "", True),
                ("schedule", "30 12 * * *", False),
            ]:
                with self.subTest(event=event, schedule=schedule):
                    script = step_script("Collect one intraday attempt").replace("${{ github.event_name }}", event)
                    result = subprocess.run(
                        ["bash", "-e", "-c", 'python3() { printf "%s\\n" "$@" > "$CAPTURE_PATH"; };\n' + script],
                        env={**os.environ, "COLLECTION_SCHEDULE": schedule, "RUNNER_TEMP": tmp, "CAPTURE_PATH": str(capture)},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = capture.read_text().splitlines()
                    self.assertEqual("--allow-outside-window" in args, immediate)
                    self.assertEqual("--collect-sample-times" in args, not immediate)
                    self.assertIn("--observation-backup-dir", args)

    def test_failed_snapshot_pushes_are_failures_not_silent_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state/site/intraday/2026-10-11").mkdir(parents=True)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_git = bin_dir / "git"
            fake_git.write_text('''#!/bin/bash
case "$*" in
  *"status --porcelain"*) echo "?? site/intraday/2026-10-11/read.json" ;;
  *"push origin"*) exit 1 ;;
  *"pull --rebase"*) exit "$REBASE_RESULT" ;;
esac
''')
            fake_git.chmod(0o755)
            for rebase_result in ("0", "1"):
                with self.subTest(rebase_result=rebase_result):
                    result = subprocess.run(
                        ["bash", "-e", "-c", step_script("Persist intraday snapshots")],
                        cwd=root,
                        env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "REBASE_RESULT": rebase_result},
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("::error::", result.stdout)
                    self.assertIn("Recover the raw observation artifact", result.stdout)


if __name__ == "__main__":
    unittest.main()
