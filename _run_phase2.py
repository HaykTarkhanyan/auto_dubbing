import logging, sys, io, pickle, re

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
for n in ("httpcore", "httpx", "google_genai", "urllib3"):
    logging.getLogger(n).setLevel(logging.WARNING)

from pipeline import run_pipeline_phase2
from modules.transcript import TranscriptSegment

# Load phase1
with open("_phase1.pkl", "rb") as f:
    phase1 = pickle.load(f)

# Parse edited translations from file
with open("translations_review.txt", "r", encoding="utf-8") as f:
    content = f.read()

edited_segments = []
for i, orig in enumerate(phase1.original_segments):
    pattern = r"--- Segment " + str(i) + r" .*?---\n  EN:.*?\n  HY: (.*?)(?:\n|$)"
    match = re.search(pattern, content)
    if match:
        edited_text = match.group(1).strip()
    else:
        edited_text = phase1.translated_segments[i].text
    edited_segments.append(TranscriptSegment(
        text=edited_text,
        start=orig.start,
        duration=orig.duration,
    ))

print(f"Loaded {len(edited_segments)} edited segments")
for i, seg in enumerate(edited_segments):
    print(f"  [{i}] {seg.text[:80]}..." if len(seg.text) > 80 else f"  [{i}] {seg.text}")

print()
print("Running phase 2...")
result = run_pipeline_phase2(phase1, edited_segments)

if result.error:
    print(f"ERROR: {result.error}")
else:
    print(f"\nOutput: {result.output_video_path}")
    print(f"Cost: ${result.cost_tracker.total_cost:.4f}")
