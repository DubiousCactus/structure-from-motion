from typing import Optional

import typer

from sfm.feature_extraction import extract_and_match_impl, extract_frames_impl

app = typer.Typer()


@app.command()
def extract_frames(video_path: str, output_folder: str):
    extract_frames_impl(video_path, output_folder)


@app.command()
def extract_and_match(
    frames_path: str,
    intrinsics_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    debug: Optional[bool] = False,
):
    extract_and_match_impl(frames_path, intrinsics_path, max_frames, debug)


if __name__ == "__main__":
    app()
