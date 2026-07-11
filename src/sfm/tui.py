from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class PairStats:
    pair_label: str = ""
    features_a: int = 0
    features_b: int = 0
    raw_matches: int = 0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    sampson_error: float = 0.0
    ransac_running: bool = False
    ransac_iter: int = 0
    ransac_best_inliers: int = 0
    ransac_best_error: float = 0.0
    done: bool = False


@dataclass
class PipelineStats:
    stage: str = "init"
    total_frames: int = 0
    frames_processed: int = 0
    features_per_frame: dict[int, int] = field(default_factory=dict)
    total_pairs: int = 0
    pairs_processed: int = 0
    per_pair: list[PairStats] = field(default_factory=list)
    bootstrap_3d_points: int = 0
    bootstrap_cameras_registered: int = 0
    pnp_cameras_registered: int = 0
    pnp_total_3d_points: int = 0
    log_messages: list[str] = field(default_factory=list)
    error: Optional[str] = None


STAGE_LABELS = {
    "init": "Initializing",
    "feature_extraction": "Feature Extraction (ORB)",
    "matching": "KNN Feature Matching",
    "ransac": "RANSAC Outlier Rejection",
    "bootstrap": "E-matrix Bootstrap & Triangulation",
    "pnp": "PnP Pose Estimation",
    "bundle_adjustment": "Bundle Adjustment",
    "done": "Complete",
}


class SfmDisplay:
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._live: Optional[Live] = None
        self.stats = PipelineStats()

    def __enter__(self) -> SfmDisplay:
        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._live:
            self._live.stop()

    def update(self) -> None:
        if self._live:
            self._live.update(self._build_layout())

    def log(self, msg: str) -> None:
        from datetime import datetime

        ts = datetime.now().strftime("%H:%M:%S")
        self.stats.log_messages.append(f"[dim]{ts}[/dim] {msg}")
        self.update()

    def set_stage(self, stage: str) -> None:
        self.stats.stage = stage
        self.update()

    def begin_extraction(self, total_frames: int) -> None:
        self.stats.total_frames = total_frames
        self.stats.frames_processed = 0
        self.stats.features_per_frame.clear()
        self.set_stage("feature_extraction")

    def update_extraction(self, frame_id: int, n_keypoints: int) -> None:
        self.stats.features_per_frame[frame_id] = n_keypoints
        self.stats.frames_processed = len(self.stats.features_per_frame)
        self.update()

    def finish_extraction(self) -> None:
        self.log(
            f"Extracted {self.stats.frames_processed} frames, "
            f"{sum(self.stats.features_per_frame.values())} total keypoints"
        )

    def begin_matching(self, total_pairs: int) -> None:
        self.stats.total_pairs = total_pairs
        self.stats.pairs_processed = 0
        self.stats.per_pair.clear()
        self.set_stage("matching")

    def update_matching(self, pair_idx: int, pair: PairStats) -> None:
        while len(self.stats.per_pair) <= pair_idx:
            self.stats.per_pair.append(PairStats())
        self.stats.per_pair[pair_idx] = pair
        self.stats.pairs_processed = sum(1 for p in self.stats.per_pair if p.done)
        self.update()

    def begin_ransac(self, pair_idx: int) -> None:
        self.set_stage("ransac")
        if pair_idx < len(self.stats.per_pair):
            self.stats.per_pair[pair_idx].ransac_running = True
            self.stats.per_pair[pair_idx].ransac_iter = 0
        self.update()

    def update_ransac(
        self, pair_idx: int, iteration: int, inliers: int, error: float
    ) -> None:
        while len(self.stats.per_pair) <= pair_idx:
            self.stats.per_pair.append(PairStats())
        p = self.stats.per_pair[pair_idx]
        p.ransac_iter = iteration
        p.ransac_best_inliers = inliers
        p.ransac_best_error = error
        self.update()

    def finish_ransac_pair(
        self, pair_idx: int, inliers: int, ratio: float, error: float
    ) -> None:
        while len(self.stats.per_pair) <= pair_idx:
            self.stats.per_pair.append(PairStats())
        p = self.stats.per_pair[pair_idx]
        p.inliers = inliers
        p.inlier_ratio = ratio
        p.sampson_error = error
        p.ransac_running = False
        p.done = True
        self.stats.pairs_processed = sum(1 for pp in self.stats.per_pair if pp.done)
        self.update()

    def set_bootstrap_result(
        self, n_3d_points: int, n_cameras: int
    ) -> None:
        self.stats.bootstrap_3d_points = n_3d_points
        self.stats.bootstrap_cameras_registered = n_cameras
        self.set_stage("bootstrap")
        self.log(f"Bootstrap: {n_cameras} cameras, {n_3d_points} 3D points")

    def update_pnp(self, cameras_registered: int, total_3d_points: int) -> None:
        self.stats.pnp_cameras_registered = cameras_registered
        self.stats.pnp_total_3d_points = total_3d_points
        self.set_stage("pnp")
        self.update()

    def finish(self) -> None:
        self.set_stage("done")
        self.log("Pipeline complete")

    def set_error(self, msg: str) -> None:
        self.stats.error = msg
        self.update()

    # ── Rendering ─────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="log", size=min(8, len(self.stats.log_messages) + 2)),
        )
        layout["header"].update(self._render_header())
        layout["body"].update(self._render_body())
        layout["log"].update(self._render_log())
        return layout

    def _render_header(self) -> Panel:
        stage_label = STAGE_LABELS.get(self.stats.stage, self.stats.stage)
        parts = [Text.assemble(("SfM Pipeline", "bold"), ("  —  ", "dim"), (stage_label, "bold cyan"))]
        if self.stats.error:
            parts.append(Text(f"\n  ✗ {self.stats.error}", style="bold red"))
        return Panel(Text.join(Text(""), parts), style="blue")

    def _render_body(self) -> Layout:
        layout = Layout()
        rows = []
        if self.stats.stage in ("feature_extraction",) or self.stats.frames_processed > 0:
            rows.append(("extraction", 7 + min(self.stats.frames_processed, 8)))
        if self.stats.total_pairs > 0:
            rows.append(("matching", 4 + min(self.stats.total_pairs, 12)))
        if self.stats.bootstrap_cameras_registered > 0 or self.stats.pnp_cameras_registered > 0:
            rows.append(("structure", 5))
        if not rows:
            rows.append(("extraction", 5))
        layout.split_column(*[Layout(name=n, size=s) for n, s in rows])
        if "extraction" in [n for n, _ in rows]:
            layout["extraction"].update(self._render_extraction())
        if "matching" in [n for n, _ in rows]:
            layout["matching"].update(self._render_matching())
        if "structure" in [n for n, _ in rows]:
            layout["structure"].update(self._render_structure())
        return layout

    def _render_extraction(self) -> Panel:
        s = self.stats
        if s.total_frames == 0:
            return Panel("[dim]Waiting for frames...[/dim]", title="Feature Extraction")
        progress = s.frames_processed / s.total_frames
        filled = int(progress * 30)
        bar = "█" * filled + "░" * (30 - filled)
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("Progress", f"[cyan]{bar}[/cyan] {s.frames_processed}/{s.total_frames}")
        if s.features_per_frame:
            sample = list(s.features_per_frame.items())[-6:]
            kp_table = Table(show_header=False, box=None, padding=(0, 2))
            kp_table.add_column("Frame", style="dim")
            kp_table.add_column("Keypoints", justify="right")
            for fid, nk in sample:
                kp_table.add_row(f"frame_{fid:04d}", str(nk))
            if len(s.features_per_frame) > 6:
                kp_table.add_row("[dim]...[/dim]", f"[dim]+{len(s.features_per_frame) - 6} more[/dim]")
            table.add_row("Sample", kp_table)
        return Panel(table, title="Feature Extraction")

    def _render_matching(self) -> Panel:
        s = self.stats
        table = Table(
            header_style="bold",
            show_lines=False,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Pair", style="bold", width=10)
        table.add_column("Feat A", justify="right", width=7)
        table.add_column("Feat B", justify="right", width=7)
        table.add_column("Raw", justify="right", width=6)
        table.add_column("Good", justify="right", width=6)
        table.add_column("Inliers", justify="right", width=8)
        table.add_column("Ratio", justify="right", width=7)
        table.add_column("Sampson ε", justify="right", width=9)
        table.add_column("RANSAC", width=18)
        for p in s.per_pair:
            if p.ransac_running:
                ransac_col = Text(
                    f"iter {p.ransac_iter}  {p.ransac_best_inliers} inl",
                    style="yellow",
                )
            elif p.done:
                ransac_col = Text("done", style="green")
            else:
                ransac_col = Text("—", style="dim")
            inlier_str = str(p.inliers) if p.done else ("—" if not p.ransac_running else str(p.ransac_best_inliers))
            ratio_str = f"{p.inlier_ratio:.0%}" if p.done else "—"
            err_str = f"{p.sampson_error:.3f}" if p.done and p.sampson_error > 0 else ("—" if not p.ransac_running else f"{p.ransac_best_error:.3f}")
            table.add_row(
                p.pair_label,
                str(p.features_a) if p.features_a else "—",
                str(p.features_b) if p.features_b else "—",
                str(p.raw_matches) if p.raw_matches else "—",
                str(p.good_matches) if p.good_matches else "—",
                inlier_str,
                ratio_str,
                err_str,
                ransac_col,
            )
        return Panel(table, title=f"Matching & RANSAC  [{s.pairs_processed}/{s.total_pairs} pairs]")

    def _render_structure(self) -> Panel:
        s = self.stats
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column(justify="right")
        cam_total = s.bootstrap_cameras_registered + s.pnp_cameras_registered
        grid.add_row("Cameras registered", f"[green]{cam_total}[/green] / {s.total_frames}")
        pts = s.pnp_total_3d_points or s.bootstrap_3d_points
        grid.add_row("3D points", f"[cyan]{pts:,}[/cyan]")
        return Panel(grid, title="Structure")

    def _render_log(self) -> Panel:
        msgs = self.stats.log_messages[-6:]
        if not msgs:
            msgs = ["[dim]No events yet...[/dim]"]
        return Panel(
            "\n".join(msgs),
            title="Log",
            border_style="dim",
        )
