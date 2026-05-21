"""YAML 配置加载。

将 configs/default.yaml 的层级结构封装成带属性访问的小型数据类,
脚本里写 `cfg.camera.width` 而不是 `cfg["camera"]["width"]`,
减少手抖出错的几率。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .paths import resolve_under_root


@dataclass
class PathsConfig:
    database: str
    data_raw: str
    data_processed: str
    data_features: str
    results_figures: str

    def database_abs(self) -> Path:
        return resolve_under_root(self.database)

    def data_raw_abs(self) -> Path:
        return resolve_under_root(self.data_raw)

    def data_processed_abs(self) -> Path:
        return resolve_under_root(self.data_processed)

    def data_features_abs(self) -> Path:
        return resolve_under_root(self.data_features)

    def results_figures_abs(self) -> Path:
        return resolve_under_root(self.results_figures)


@dataclass
class CameraConfig:
    model: str
    width: int
    height: int
    fps: float
    dtype: str
    endian: str
    header_offset: int
    temperature_scale: float
    dx_mm_per_pixel: Optional[float]
    dy_mm_per_pixel: Optional[float]


@dataclass
class ExperimentConfig:
    name: str
    powder_material: str
    substrate_material: str
    laser_power_W: float
    scan_speed_mm_per_min: float
    powder_feed_rate_g_per_min: float
    hatch_spacing_mm: float
    B_levels_mT: List[float]


@dataclass
class ProcessingConfig:
    high_temp_threshold_C: float
    gaussian_sigma_px: float
    log_every_frames: int
    max_frames: Optional[int]


@dataclass
class FormatProbeConfig:
    header_offset_candidates: List[int]
    sample_frames: List[Any]  # mix of int and str like "middle"


@dataclass
class LoggingConfig:
    level: str
    format: str


@dataclass
class AppConfig:
    paths: PathsConfig
    camera: CameraConfig
    experiment: ExperimentConfig
    processing: ProcessingConfig
    format_probe: FormatProbeConfig
    logging: LoggingConfig
    source_path: Path
    raw: Dict[str, Any] = field(default_factory=dict)


def _required(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key '{key}' under {where} in config")
    return d[key]


def load_config(config_path: Union[str, Path] = "configs/default.yaml") -> AppConfig:
    """从 YAML 加载配置并校验关键字段。

    Parameters
    ----------
    config_path : str | Path
        相对项目根目录或绝对路径。默认 'configs/default.yaml'。

    Returns
    -------
    AppConfig
        结构化配置对象。
    """
    abs_path = resolve_under_root(config_path)
    if not abs_path.exists():
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    with abs_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    paths_d = _required(raw, "paths", "config root")
    cam_d = _required(raw, "camera", "config root")
    exp_d = _required(raw, "experiment", "config root")
    proc_d = _required(raw, "processing", "config root")
    fp_d = _required(raw, "format_probe", "config root")
    log_d = _required(raw, "logging", "config root")

    paths = PathsConfig(
        database=_required(paths_d, "database", "paths"),
        data_raw=_required(paths_d, "data_raw", "paths"),
        data_processed=_required(paths_d, "data_processed", "paths"),
        data_features=_required(paths_d, "data_features", "paths"),
        results_figures=_required(paths_d, "results_figures", "paths"),
    )
    camera = CameraConfig(
        model=_required(cam_d, "model", "camera"),
        width=int(_required(cam_d, "width", "camera")),
        height=int(_required(cam_d, "height", "camera")),
        fps=float(_required(cam_d, "fps", "camera")),
        dtype=str(_required(cam_d, "dtype", "camera")),
        endian=str(_required(cam_d, "endian", "camera")),
        header_offset=int(_required(cam_d, "header_offset", "camera")),
        temperature_scale=float(_required(cam_d, "temperature_scale", "camera")),
        dx_mm_per_pixel=cam_d.get("dx_mm_per_pixel"),
        dy_mm_per_pixel=cam_d.get("dy_mm_per_pixel"),
    )
    experiment = ExperimentConfig(
        name=str(_required(exp_d, "name", "experiment")),
        powder_material=str(_required(exp_d, "powder_material", "experiment")),
        substrate_material=str(_required(exp_d, "substrate_material", "experiment")),
        laser_power_W=float(_required(exp_d, "laser_power_W", "experiment")),
        scan_speed_mm_per_min=float(_required(exp_d, "scan_speed_mm_per_min", "experiment")),
        powder_feed_rate_g_per_min=float(_required(exp_d, "powder_feed_rate_g_per_min", "experiment")),
        hatch_spacing_mm=float(_required(exp_d, "hatch_spacing_mm", "experiment")),
        B_levels_mT=[float(x) for x in _required(exp_d, "B_levels_mT", "experiment")],
    )
    processing = ProcessingConfig(
        high_temp_threshold_C=float(_required(proc_d, "high_temp_threshold_C", "processing")),
        gaussian_sigma_px=float(_required(proc_d, "gaussian_sigma_px", "processing")),
        log_every_frames=int(_required(proc_d, "log_every_frames", "processing")),
        max_frames=proc_d.get("max_frames"),
    )
    format_probe = FormatProbeConfig(
        header_offset_candidates=[int(x) for x in _required(fp_d, "header_offset_candidates", "format_probe")],
        sample_frames=list(_required(fp_d, "sample_frames", "format_probe")),
    )
    logging_cfg = LoggingConfig(
        level=str(_required(log_d, "level", "logging")),
        format=str(_required(log_d, "format", "logging")),
    )

    return AppConfig(
        paths=paths,
        camera=camera,
        experiment=experiment,
        processing=processing,
        format_probe=format_probe,
        logging=logging_cfg,
        source_path=abs_path,
        raw=raw,
    )
