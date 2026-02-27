import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

LALAL_API_BASE = "https://www.lalal.ai/api/v1"


def separate_vocals_demucs(audio_path: str, output_dir: str) -> str:
    """Separate vocals using demucs (htdemucs model).

    Returns the path to the no-vocals (accompaniment) track.
    """
    audio_name = Path(audio_path).stem
    output_dir = str(Path(output_dir))

    logger.info("Running vocal separation with demucs...")
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-o", output_dir,
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs vocal separation failed: {result.stderr}")

    # Demucs output structure: {output_dir}/htdemucs/{audio_name}/no_vocals.wav
    no_vocals_path = Path(output_dir) / "htdemucs" / audio_name / "no_vocals.wav"
    if not no_vocals_path.exists():
        raise FileNotFoundError(
            f"Demucs output not found at {no_vocals_path}. "
            f"Contents: {list((Path(output_dir) / 'htdemucs').rglob('*')) if (Path(output_dir) / 'htdemucs').exists() else 'htdemucs dir missing'}"
        )

    logger.info(f"Vocal separation complete: {no_vocals_path}")
    return str(no_vocals_path)


def separate_vocals_mdx(audio_path: str, output_dir: str, model: str = "UVR_MDXNET_KARA_2.onnx") -> str:
    """Separate vocals using audio-separator with MDX-Net models.

    Returns the path to the instrumental (no-vocals) track.
    """
    from audio_separator.separator import Separator

    output_dir = str(Path(output_dir))

    logger.info(f"Running vocal separation with MDX-Net ({model})...")
    sep = Separator(output_dir=output_dir)
    sep.load_model(model)
    output_files = sep.separate(audio_path)

    # audio-separator returns [primary (vocals), secondary (instrumental)]
    if len(output_files) < 2:
        raise RuntimeError(f"MDX-Net separation returned unexpected output: {output_files}")

    instrumental_path = Path(output_files[1])
    # sep.separate() may return just filenames; resolve relative to output_dir
    if not instrumental_path.is_absolute():
        instrumental_path = Path(output_dir) / instrumental_path
    if not instrumental_path.exists():
        raise FileNotFoundError(f"MDX-Net instrumental output not found at {instrumental_path}")

    logger.info(f"Vocal separation complete: {instrumental_path}")
    return str(instrumental_path)


def separate_vocals_lalal(audio_path: str, output_dir: str, api_key: str) -> str:
    """Separate vocals using LALAL.AI API.

    Returns the path to the instrumental (no-vocals) track.
    """
    import json

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    headers = {"X-License-Key": api_key}

    # Step 1: Upload the audio file
    filename = Path(audio_path).name
    logger.info(f"LALAL.AI: Uploading {filename}...")

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    upload_req = urllib.request.Request(
        f"{LALAL_API_BASE}/upload/",
        data=audio_data,
        headers={
            **headers,
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(upload_req) as resp:
        upload_result = json.loads(resp.read().decode())

    source_id = upload_result.get("id")
    if not source_id:
        raise RuntimeError(f"LALAL.AI upload failed: {upload_result}")
    logger.info(f"LALAL.AI: Uploaded, source_id={source_id}")

    # Step 2: Start stem separation (vocals preset → get back instrumental)
    split_payload = json.dumps({
        "source_id": source_id,
        "presets": {
            "stem": "vocals",
            "splitter": "orion",
            "encoder_format": "wav",
        },
    }).encode()

    split_req = urllib.request.Request(
        f"{LALAL_API_BASE}/split/stem_separator/",
        data=split_payload,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(split_req) as resp:
        split_result = json.loads(resp.read().decode())

    logger.info(f"LALAL.AI split response: {split_result}")

    # The split response uses the source_id as task_id
    task_id = split_result.get("task_id") or split_result.get("id") or source_id
    logger.info(f"LALAL.AI: Separation started, task_id={task_id}")

    # Step 3: Poll for completion
    max_wait = 300  # 5 minutes max
    poll_interval = 3
    elapsed = 0
    task = None

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        check_payload = json.dumps({"task_ids": [task_id]}).encode()
        check_req = urllib.request.Request(
            f"{LALAL_API_BASE}/check/",
            data=check_payload,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(check_req) as resp:
            check_result = json.loads(resp.read().decode())

        logger.info(f"LALAL.AI check response: {check_result}")

        # Handle different response formats
        # API returns: {"result": {"<task_id>": {..., "status": "progress"|"success"|"error", ...}}}
        if isinstance(check_result, dict) and "result" in check_result:
            task = check_result["result"].get(task_id, {})
        elif isinstance(check_result, dict) and "tasks" in check_result:
            task = check_result["tasks"].get(task_id, {})
        elif isinstance(check_result, dict):
            task = check_result
        else:
            raise RuntimeError(f"LALAL.AI check unexpected format: {check_result}")

        task_status = task.get("status", "")

        if task_status == "success":
            logger.info(f"LALAL.AI: Separation complete ({elapsed}s)")
            break
        elif task_status == "error":
            raise RuntimeError(f"LALAL.AI separation error: {task.get('error')}")
        else:
            progress = task.get("progress", 0)
            logger.info(f"LALAL.AI: Processing... {progress}% ({elapsed}s)")
    else:
        raise RuntimeError(f"LALAL.AI separation timed out after {max_wait}s")

    # Step 4: Find and download the instrumental (back) track
    # Tracks may be at task["result"]["tracks"] or task["tracks"]
    task_result = task.get("result", task)
    tracks = task_result.get("tracks", [])
    # tracks could be a list or a dict
    if isinstance(tracks, dict):
        tracks = list(tracks.values())

    back_url = None
    for track in tracks:
        track_type = track.get("type", "")
        if track_type == "back" or "instrumental" in track.get("label", "").lower() or "no_vocal" in track.get("label", "").lower():
            back_url = track.get("url") or track.get("download_url")
            break

    if not back_url:
        raise RuntimeError(f"LALAL.AI: No instrumental track found in response: {tracks}")

    output_path = output_dir_path / "no_vocals.wav"
    logger.info(f"LALAL.AI: Downloading instrumental track...")
    urllib.request.urlretrieve(back_url, str(output_path))

    logger.info(f"Vocal separation complete: {output_path}")

    # Cleanup: delete from LALAL.AI storage
    try:
        del_payload = json.dumps({"source_ids": [source_id]}).encode()
        del_req = urllib.request.Request(
            f"{LALAL_API_BASE}/delete/",
            data=del_payload,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(del_req)
    except Exception:
        pass  # Non-critical

    return str(output_path)


def separate_vocals(audio_path: str, output_dir: str, method: str = "demucs", api_key: str = "") -> str:
    """Separate vocals from audio.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory for output files.
        method: "demucs", "mdx", or "lalal" (LALAL.AI API).
        api_key: API key (required for "lalal" method).

    Returns the path to the no-vocals (accompaniment) track.
    """
    if method == "lalal":
        if not api_key:
            raise ValueError("LALAL_API_KEY is required for LALAL.AI vocal separation")
        return separate_vocals_lalal(audio_path, output_dir, api_key)
    elif method == "mdx":
        return separate_vocals_mdx(audio_path, output_dir)
    else:
        return separate_vocals_demucs(audio_path, output_dir)
