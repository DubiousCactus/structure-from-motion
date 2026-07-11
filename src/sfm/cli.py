import os
from typing import Optional

import typer

from sfm.feature_extraction import extract_and_match_impl, extract_frames_impl
from sfm.tui import SfmDisplay

app = typer.Typer()


@app.command()
def extract_frames(video_path: str, output_folder: str):
    with SfmDisplay() as display:
        extract_frames_impl(video_path, output_folder, display=display)


@app.command()
def extract_and_match(
    frames_path: str,
    intrinsics_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    debug: Optional[bool] = False,
    orb_features: int = 2000,
    lowe_ratio: float = 0.8,
):
    with SfmDisplay() as display:
        try:
            extract_and_match_impl(
                frames_path,
                intrinsics_path,
                max_frames,
                debug,
                display=display,
                orb_features=orb_features,
                lowe_ratio=lowe_ratio,
            )
        except Exception as e:
            display.set_error(str(e))
            display.log(f"[red]Error: {e}[/red]")
            raise


@app.command(name="extract-intrinsics")
def extract_intrinsics(
    images_path: str,
    output: str = "intrinsics.npz",
):
    from rich.console import Console
    from rich.table import Table

    from sfm.intrinsics import compute_intrinsics

    console = Console()

    image_paths = sorted(
        [
            os.path.join(images_path, f)
            for f in os.listdir(images_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_path}")

    db = compute_intrinsics(image_paths)

    table = Table(title=f"Camera Intrinsics ({db.num_images} images, {db.num_cameras} camera(s))")
    table.add_column("Image", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("K", style="green")
    table.add_column("Camera ID", justify="right", style="magenta")

    for basename, cid in sorted(db.camera_ids.items()):
        K = db.cameras[cid]
        w, h = db.image_sizes[basename]
        K_str = f"[{K[0,0]:.1f} 0 {K[0,2]:.1f}; 0 {K[1,1]:.1f} {K[1,2]:.1f}; 0 0 1]"
        table.add_row(basename, f"{w}x{h}", K_str, str(cid))

    console.print(table)

    cam_table = Table(title="Unique Cameras")
    cam_table.add_column("ID", justify="right", style="magenta")
    cam_table.add_column("K", style="green")
    for i, K in enumerate(db.cameras):
        K_str = f"[{K[0,0]:.1f} 0 {K[0,2]:.1f}; 0 {K[1,1]:.1f} {K[1,2]:.1f}; 0 0 1]"
        cam_table.add_row(str(i), K_str)
    console.print(cam_table)

    K_avg = db.average_K
    K_avg_str = f"[{K_avg[0,0]:.1f} 0 {K_avg[0,2]:.1f}; 0 {K_avg[1,1]:.1f} {K_avg[1,2]:.1f}; 0 0 1]"
    console.print(f"\nAverage K = [green]{K_avg_str}[/green]")
    console.print(f"Saved camera database → [cyan]{output}[/cyan]")
    db.save(output)


if __name__ == "__main__":
    app()
