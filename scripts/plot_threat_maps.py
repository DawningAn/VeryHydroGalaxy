import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
内置配置：无需传参即可运行预览图。
根据你的文件路径设置为 HydroBASINS 亚洲 5 级 (lev05)。
如需叠加河网，填入 RIVERS_PATH。
"""
DEFAULT_CONFIG = {
    "HYDRO_PATH": r"E:\HydroSHEDS\hybas_as_lev01-12_v1c\hybas_as_lev05_v1c.shp",
    "RIVERS_PATH": r"E:\HydroSHEDS\HydroRIVERS_v10_as_shp\HydroRIVERS_v10_as_shp",
    "OUTPUT_PATH": "outputs/preview_basins.png",
    "ATTR": "SUB_AREA",
    "ID_COLUMN": "HYBAS_ID",
}

# 河网绘制样式：接近附图的浅灰粗线
RIVER_STYLE = {
    "color": "#C7CED6",  # 浅灰
    "linewidth": 2.0,
    "alpha": 0.95,
}

def read_and_join(hydro_units_path: str, species_csv: str, id_column: str):
    """Read hydrologic units and species table, join by basin id.

    The species table must have columns: id_column, and VU, EN, CR counts.
    """
    hu = gpd.read_file(hydro_units_path)
    # Normalize id column names
    if id_column not in hu.columns:
        raise ValueError(f"'{id_column}' not found in hydrologic units shapefile columns: {list(hu.columns)}")

    sp = pd.read_csv(species_csv)
    if id_column not in sp.columns:
        raise ValueError(f"'{id_column}' not found in species CSV columns: {list(sp.columns)}")
    for col in ["VU", "EN", "CR"]:
        if col not in sp.columns:
            raise ValueError(f"Species CSV must contain '{col}' column.")

    merged = hu.merge(sp[[id_column, "VU", "EN", "CR"]], on=id_column, how="left").fillna({"VU": 0, "EN": 0, "CR": 0})
    return merged


def add_north_arrow(ax, size=0.12):
    """Add a simple north arrow with 'N' in axes coordinates."""
    ax.annotate(
        "N",
        xy=(0.95, 0.92),
        xytext=(0.95, 0.78),
        textcoords=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
    )


def plot_category(ax, gdf, rivers_gdf, category, vmax=None, cmap="RdYlBu_r"):
    """Plot a single category panel with colorbar and rivers overlay."""
    if vmax is None:
        vmax = int(np.ceil(gdf[category].quantile(0.99)))

    gdf.plot(
        column=category,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        edgecolor="none",
    )

    if rivers_gdf is not None and not rivers_gdf.empty:
        rivers_gdf.plot(ax=ax, **RIVER_STYLE)

    ax.set_axis_off()
    add_north_arrow(ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm._A = []
    cax = inset_axes(ax, width="45%", height="4%", loc="lower left", borderpad=1.2)
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Species Richness")


def ensure_projected(gdf, target_crs="EPSG:3857"):
    """Project GeoDataFrame to target CRS if needed."""
    try:
        return gdf.to_crs(target_crs)
    except Exception:
        return gdf


def _resolve_shp_path(path_like: str) -> Path:
    """Resolve a shapefile path: accept file or directory containing .shp.

    If a directory is provided, pick the first .shp file (prefer names containing
    'HydroRIVERS').
    """
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() == ".shp":
        return p
    if p.is_dir():
        candidates = sorted(list(p.glob("*HydroRIVERS*.shp")))
        if not candidates:
            candidates = sorted(list(p.glob("*.shp")))
        return candidates[0] if candidates else p
    return p


def load_rivers(rivers_path: str):
    """Safely load rivers shapefile; return None on failure.

    Accepts a .shp file or a directory containing HydroRIVERS .shp.
    """
    if not rivers_path:
        return None
    try:
        shp = _resolve_shp_path(rivers_path)
        if shp.is_file() and shp.suffix.lower() == ".shp":
            return gpd.read_file(shp)
        else:
            print(f"No .shp found under '{rivers_path}', skip rivers overlay.")
            return None
    except Exception as e:
        print(f"Failed to read rivers from '{rivers_path}': {e}")
        return None


def select_trunk_rivers(rivers_gdf: gpd.GeoDataFrame):
    """按用户要求仅保留 ORD_FLOW ∈ {2,3,4} 的河段。

    该筛选与 HydroRIVERS 的 `ORD_FLOW` 字段直接匹配，确保只显示主要干流。
    """
    if rivers_gdf is None or rivers_gdf.empty:
        return rivers_gdf

    try:
        gdf = rivers_gdf.copy()
        ord_flow = pd.to_numeric(gdf.get("ORD_FLOW", None), errors="coerce")
        mask = ord_flow.isin([2, 3, 4])
        gdf = gdf[mask]
        print(f"Selected {len(gdf)} rivers with ORD_FLOW in [2,3,4] out of {len(rivers_gdf)}")
        return gdf
    except Exception as e:
        print(f"Error selecting trunk rivers by ORD_FLOW: {e}")
        return rivers_gdf


def plot_attr_only(ax, gdf, rivers_gdf, attr: str, cmap="Spectral_r", vmax=None):
    """Plot hydrologic units colored by a chosen attribute.

    If attr is missing, compute area (km²) and use it.
    """
    # If attribute not available, compute polygon area as fallback
    if attr not in gdf.columns:
        # compute area in km² (approximate under Web Mercator)
        gdf["AREA_KM2"] = gdf.geometry.area / 1_000_000.0
        attr = "AREA_KM2"

    if vmax is None:
        vmax = float(np.nanpercentile(gdf[attr], 99))

    gdf.plot(column=attr, ax=ax, cmap=cmap, vmin=0, vmax=vmax, edgecolor="none")
    if rivers_gdf is not None and not rivers_gdf.empty:
        rivers_gdf.plot(ax=ax, **RIVER_STYLE)

    ax.set_axis_off()
    add_north_arrow(ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm._A = []
    cax = inset_axes(ax, width="45%", height="4%", loc="lower left", borderpad=1.2)
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(attr)


def main(hydro_units_path: str, species_csv: str = None, rivers_path: str = None, id_column: str = "HYBAS_ID", output_path: str = "outputs/figure_s3.png", attr: str = "SUB_AREA"):
    # Read hydrologic units
    gdf = gpd.read_file(hydro_units_path)

    # Optional rivers with trunk selection
    rivers_gdf = None
    if rivers_path:
        rivers_gdf = load_rivers(rivers_path)
        rivers_gdf = select_trunk_rivers(rivers_gdf)

    # Reproject to Web Mercator for easy plotting
    target_crs = "EPSG:3857"
    gdf = ensure_projected(gdf, target_crs)
    if rivers_gdf is not None:
        rivers_gdf = ensure_projected(rivers_gdf, target_crs)

    # If no species CSV is provided, draw a single preview map colored by attribute
    if species_csv is None or len(str(species_csv).strip()) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        plot_attr_only(ax, gdf, rivers_gdf, attr)
        ax.set_title(f"HydroBasins colored by '{attr}'", fontsize=11)

        out_path = Path(output_path if output_path else "outputs/preview_basins.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 同时输出 PNG(位图) 与 SVG(矢量)，便于屏幕查看与放大打印
        fig.set_size_inches(10, 8)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        out_svg = out_path.with_suffix(".svg")
        try:
            fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
            print(f"Saved preview map to {out_path} and {out_svg}")
        except Exception:
            print(f"Saved preview map to {out_path}. SVG export failed (likely missing backend)")
        return

    # Otherwise join with species table and draw three panels
    gdf = read_and_join(hydro_units_path, species_csv, id_column)
    gdf = ensure_projected(gdf, target_crs)
    fig, axes = plt.subplots(3, 1, figsize=(8, 14), constrained_layout=True)
    categories = ["VU", "EN", "CR"]
    labels = ["a", "b", "c"]

    for ax, cat, lab in zip(axes, categories, labels):
        plot_category(ax, gdf, rivers_gdf, cat)
        ax.text(0.01, 0.95, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.set_title(f"Species Richness ({cat})", fontsize=11)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 位图 + 矢量双份输出，保证清晰度
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    out_svg = out_path.with_suffix(".svg")
    try:
        fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to {out_path} and {out_svg}")
    except Exception:
        print(f"Saved figure to {out_path}. SVG export failed (likely missing backend)")


if __name__ == "__main__":
    # 如果提供了命令行参数，则走参数模式；否则使用内置 DEFAULT_CONFIG。
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Plot fish threatened level maps per hydrologic unit (or preview HydroBasins coloring without CSV)")
        parser.add_argument("--hydro", required=True, help="Path to hydrologic units shapefile (e.g., HydroBASINS for China)")
        parser.add_argument("--species", help="Path to species counts CSV with columns: id, VU, EN, CR (optional)")
        parser.add_argument("--rivers", help="Optional rivers shapefile (e.g., HydroRIVERS)")
        parser.add_argument("--id_col", default="HYBAS_ID", help="Join key column name present in both files")
        parser.add_argument("--attr", default="SUB_AREA", help="Attribute column to color basins when species CSV is not provided (fallback to area)")
        parser.add_argument("--output", default="outputs/figure_s3.png", help="Output image path (or preview path)")

        args = parser.parse_args()
        main(
            hydro_units_path=args.hydro,
            species_csv=args.species,
            rivers_path=args.rivers,
            id_column=args.id_col,
            output_path=args.output,
            attr=args.attr,
        )
    else:
        main(
            hydro_units_path=DEFAULT_CONFIG["HYDRO_PATH"],
            species_csv=None,
            rivers_path=DEFAULT_CONFIG["RIVERS_PATH"],
            id_column=DEFAULT_CONFIG["ID_COLUMN"],
            output_path=DEFAULT_CONFIG["OUTPUT_PATH"],
            attr=DEFAULT_CONFIG["ATTR"],
        )