from pathlib import Path
import geopandas as gpd
import pandas as pd

RIVERS_DIR_OR_FILE = r"E:\HydroSHEDS\HydroRIVERS_v10_as_shp\HydroRIVERS_v10_as_shp"

def resolve_shp(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() == ".shp":
        return p
    if p.is_dir():
        cands = sorted(p.glob("*HydroRIVERS*.shp")) or sorted(p.glob("*.shp"))
        return cands[0] if cands else p
    return p

def main():
    shp = resolve_shp(RIVERS_DIR_OR_FILE)
    print("Resolved:", shp)
    gdf = gpd.read_file(shp)
    print("Rows:", len(gdf))
    print("CRS:", gdf.crs)
    print("Columns:", list(gdf.columns))
    # Basic stats
    if "MAIN_RIV" in gdf.columns:
        print("MAIN_RIV==1:", int((gdf["MAIN_RIV"] == 1).sum()))
    if "ORD_FLOW" in gdf.columns:
        ord_flow = pd.to_numeric(gdf["ORD_FLOW"], errors="coerce")
        q75 = float(ord_flow.quantile(0.75))
        q90 = float(ord_flow.quantile(0.90))
        thr = int(max(6, q75))
        print("ORD_FLOW quantiles (75,90):", q75, q90)
        print("ORD_FLOW>=", thr, ":", int((ord_flow >= thr).sum()))
    if "LENGTH_KM" in gdf.columns:
        ln = pd.to_numeric(gdf["LENGTH_KM"], errors="coerce")
        print("LENGTH_KM quantiles (75,90,95):", float(ln.quantile(0.75)), float(ln.quantile(0.90)), float(ln.quantile(0.95)))
        print("LENGTH_KM>=50:", int((ln >= 50).sum()))
    if "CATCH_SKM" in gdf.columns:
        catch = pd.to_numeric(gdf["CATCH_SKM"], errors="coerce")
        print("CATCH_SKM quantiles (90,95,99):", float(catch.quantile(0.90)), float(catch.quantile(0.95)), float(catch.quantile(0.99)))

if __name__ == "__main__":
    main()