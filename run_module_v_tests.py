import json
import sys
import time
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


def _write_summary_image(path: Path, report: dict) -> None:
    width, height = 900, 520
    img = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    lines = [
        "Module V Test Summary",
        "",
        f"timestamp: {report['timestamp']}",
        f"total: {report['total']}",
        f"passed: {report['passed']}",
        f"failed: {report['failed']}",
        f"errors: {report['errors']}",
        f"skipped: {report['skipped']}",
        "",
        f"result: {report['result']}",
    ]
    x, y = 30, 30
    for line in lines:
        draw.text((x, y), line, fill=(20, 20, 20))
        y += 28
    img.save(path)


def run() -> int:
    base = Path(__file__).parent
    out_dir = base / "out" / "module_v"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = unittest.TestLoader()
    suite = loader.discover(str(base / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "result": "passed" if result.wasSuccessful() else "failed",
    }

    (out_dir / "test_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "test_report.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in report.items()),
        encoding="utf-8",
    )
    _write_summary_image(out_dir / "test_summary.png", report)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run())

