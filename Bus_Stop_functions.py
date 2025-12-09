import math
from dataclasses import dataclass
from typing import Tuple, Optional
from shapely.ops import unary_union, linemerge, snap
from shapely.geometry import LineString, MultiLineString, Point
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans

def _merged_route_line(roads: gpd.GeoDataFrame,S = 100)-> list:
    u = unary_union(roads.geometry.values)
    u = snap(u, u, S)       # cierra micro-gaps reales
    merged = linemerge(u)
    parts = merged.geoms if isinstance(merged, MultiLineString) else [merged]
    lines = sorted(list(merged.geoms), key=lambda g: (g.bounds[1], g.bounds[0]))
    return lines

def points_every(line, step) -> list:
    L = line.length
    d = 0.0
    pts = []
    while d <= L:
        pts.append(line.interpolate(d))
        d += step
    return pts

def _snap_point_to_line_and_m(point: Point, line) -> Tuple[Point, float]:
    """
    Snaps a point to a LineString or MultiLineString and returns the snapped point
    and the m-value (distance along the line from the start).
    """
    from shapely.ops import nearest_points

    if isinstance(line, LineString):
        snapped = nearest_points(point, line)[0]
        coords = list(line.coords)
        dists = [0.0]
        for i in range(1, len(coords)):
            dists.append(dists[-1] + Point(coords[i-1]).distance(Point(coords[i])))
        total_len = dists[-1]
        min_dist = float("inf"); seg_idx = 0
        for i in range(1, len(coords)):
            seg = LineString([coords[i-1], coords[i]])
            d = snapped.distance(seg)
            if d < min_dist:
                min_dist = d; seg_idx = i
        a = Point(coords[seg_idx-1]); b = Point(coords[seg_idx])
        ab = np.array([b.x - a.x, b.y - a.y])
        ap = np.array([snapped.x - a.x, snapped.y - a.y])
        seg_len = np.linalg.norm(ab)
        t = 0.0 if seg_len == 0 else np.clip(np.dot(ap, ab) / (seg_len**2), 0, 1)
        m = dists[seg_idx-1] + t * seg_len
        return snapped, m

    elif isinstance(line, MultiLineString):
        min_dist = float("inf")
        best_snapped = None
        best_m = 0.0
        total_len_so_far = 0.0

        for segment in line.geoms:
            try:
                snapped_seg, m_seg = _snap_point_to_line_and_m(point, segment)
                dist_to_segment = point.distance(snapped_seg)

                if dist_to_segment < min_dist:
                    min_dist = dist_to_segment
                    best_snapped = snapped_seg
                    # Calculate m-value along the *entire* MultiLineString
                    best_m = total_len_so_far + m_seg
            except Exception:
                # Handle cases where snapping to a small segment might fail
                pass
            total_len_so_far += segment.length

        if best_snapped is None:
             # If snapping failed for all segments, return a default or raise error
             # For now, return the original point and 0 m-value as a fallback
             return point, 0.0 # Or raise an informative error

        return best_snapped, best_m

    else:
        raise TypeError("Input geometry must be a LineString or MultiLineString")


# --- Candidate generation ---
#espaciar cada 3 km

def make_candidates_along_route(roads: gpd.GeoDataFrame, spacing_m: int = 3000):
    route_line_segments = _merged_route_line(roads)
    pts = []
    Distancia = 0
    for tramo in route_line_segments:
        if Distancia > spacing_m:
          puntos = points_every(tramo, spacing_m)
          for punto in puntos:
              pts.append(punto)
          Distancia = 0
        else:
          Distancia += tramo.length

    return gpd.GeoDataFrame({"cand_id": range(len(pts))}, geometry=pts, crs=roads.crs)
                      ##################################################################
                      ###################### hasta aca funciona  #######################
                      ##################################################################
def expand_candidates(candidates, roads, expand_m: int = 500):
    route_line_segments = _merged_route_line(roads)
    new_pts = []
    # Consider a small buffer for proximity check
    buffer_distance = 1.0 # meters, adjust as needed

    for p in candidates.geometry:
        for tramo in route_line_segments:
            # Check if the point is within a small buffer of the line segment
            if p.within(tramo.buffer(buffer_distance)):
                _, m = _snap_point_to_line_and_m(p, tramo)
                for delta in (-expand_m, 0, +expand_m):
                    md = max(0, min(tramo.length, m + delta))
                    new_pts.append(tramo.interpolate(md))
                # Once a segment is found, no need to check others for this point
                break

    return gpd.GeoDataFrame(geometry=new_pts, crs=roads.crs).drop_duplicates(ignore_index=True)



# --- Clustering ---

"""
def kmeans_centers(demand_pts: gpd.GeoDataFrame, k: int):
    X = np.c_[demand_pts.geometry.x, demand_pts.geometry.y]
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    counts = pd.Series(labels).value_counts().sort_index().to_numpy()
    return gpd.GeoDataFrame({"weight": counts}, geometry=[Point(xy) for xy in centers], crs=demand_pts.crs)
"""
def kmeans_centers(demand_pts: gpd.GeoDataFrame, k: int):
    X = np.c_[demand_pts.geometry.x, demand_pts.geometry.y]
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # cuenta puntos en cada cluster, incluyendo los vacíos
    counts = np.bincount(labels, minlength=k)

    return gpd.GeoDataFrame(
        {"weight": counts},
        geometry=[Point(xy) for xy in centers],
        crs=demand_pts.crs
    )


# --- WPDM scoring ---
####### REVISAR
####### Por que proyecta en la ruta las ciudades, los candidatos ya van a estar en la ruta
import numpy as np
import pandas as pd
from pyproj import CRS


def wpdm_scores(candidates, centers, shortlist_k: int = 3, rank_weights=(1.0, 0.5, 0.2)):
    """
    candidates: GeoSeries (puntos) o array de floats (distancias euclídeas si no es Geo)
    centers: GeoDataFrame con columnas: geometry (Point) y weight (float)
    """
    shortlist_k = int(shortlist_k)
    if shortlist_k < 1:
        raise ValueError("shortlist_k debe ser >= 1")

    # Asegurar pesos suficientes (relleno con 0: más allá de los 3 primeros no suma)
    rweights = np.array(rank_weights, dtype=float)
    if len(rweights) < shortlist_k:
        rweights = np.pad(rweights, (0, shortlist_k - len(rweights)), constant_values=0.0)

    scores = np.zeros(len(candidates), dtype=float)

    use_geo = hasattr(candidates, "distance") and hasattr(centers, "geometry")
    center_weights = np.asarray(centers["weight"], dtype=float)

    for i, w in enumerate(center_weights):
        if use_geo:
            dists = np.asarray(candidates.distance(centers.geometry.iloc[i]))
        else:
            # si no es Geo, asumir candidatos y centers son arrays/series numéricas comparables
            dists = np.abs(np.asarray(candidates) - np.asarray(centers.iloc[i]))

        # Elegir los k más cercanos y ordenarlos por distancia
        k = min(shortlist_k, len(dists))
        idx = np.argpartition(dists, k-1)[:k]
        idx = idx[np.argsort(dists[idx])]
        scores[idx] += w * rweights[:k]

    return pd.DataFrame({"score": scores})

def select_top_n_with_min_spacing(
    candidates: gpd.GeoDataFrame,
    scores: pd.DataFrame,
    n: int,
    min_spacing_m: float = 100.0,
    score_col: str = "score",
    metric_crs: str = "EPSG:32721",  # UTM 21S (Uruguay): distancias en metros
):
    # --- Validaciones ---
    if not isinstance(candidates, gpd.GeoDataFrame):
        raise TypeError("candidates debe ser un GeoDataFrame.")
    if score_col not in scores.columns:
        raise ValueError(f"'{score_col}' no existe en scores.")
    if len(scores) != len(candidates):
        raise ValueError("candidates y scores deben tener el mismo largo (1:1).")
    if candidates.crs is None:
        raise ValueError("El GeoDataFrame no tiene CRS. Define uno (WGS84, etc.).")

    # Deben ser puntos
    if not (candidates.geometry.geom_type == "Point").all():
        raise ValueError("La geometría debe ser de tipo Point en todas las filas.")

    # --- Trabajar en metros ---
    crs_obj = CRS.from_user_input(candidates.crs)
    g = candidates
    if crs_obj.is_geographic:
        # reproyectamos a CRS en metros (por defecto UTM 21S)
        g = candidates.to_crs(metric_crs)

    # --- Selección ávida por score con espaciamiento mínimo ---
    order = (
        pd.DataFrame({"idx": np.arange(len(g)), "score": scores[score_col].to_numpy()})
        .dropna(subset=["score"])
        .sort_values("score", ascending=False)["idx"]
        .to_list()
    )

    chosen = []
    for idx in order:
        if len(chosen) >= n:
            break
        geom = g.geometry.iloc[idx]
        # distancia mínima al conjunto ya elegido
        if not chosen:
            chosen.append(idx)
            continue
        dmin = min(geom.distance(g.geometry.iloc[j]) for j in chosen)
        if dmin >= float(min_spacing_m):
            chosen.append(idx)

    out = candidates.iloc[chosen].copy()
    out[score_col] = scores.iloc[chosen][score_col].to_numpy()
    return out.sort_values(score_col, ascending=False).reset_index(drop=True)

RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45,
    10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
}

def build_pairwise_matrix(items, comparisons, check_scale=True):
    labels = list(items)
    n = len(labels)
    idx = {name: k for k, name in enumerate(labels)}
    A = np.ones((n, n), dtype=float)
    for (i, j), v in comparisons.items():
        if i not in idx or j not in idx:
            raise ValueError(f"Unknown item in comparison: {(i,j)}")
        if v <= 0:
            raise ValueError("Comparison values must be positive.")
        if check_scale and not (1/9 <= v <= 9):
            print(f"Warning: value {v} for ({i},{j}) is outside Saaty 1..9 (or reciprocal). Proceeding.")
        A[idx[i], idx[j]] = v
        A[idx[j], idx[i]] = 1.0 / v
    np.fill_diagonal(A, 1.0)
    return A, labels

def ahp_weights(A, method="eigen"):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    n = A.shape[0]

    if method == "eigen":
        vals, vecs = np.linalg.eig(A)
        idx = np.argmax(vals.real)
        w = vecs[:, idx].real
        w = np.abs(w)
    elif method == "geom":
        w = np.prod(A, axis=1) ** (1.0 / n)
    else:
        raise ValueError("method must be 'eigen' or 'geom'")

    w = w / w.sum()

    Aw = A @ w
    lambda_max = float((Aw / w).mean())
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, RI_TABLE[max(k for k in RI_TABLE if k <= n)])
    CR = CI / RI if RI > 0 else 0.0
    return {"weights": w, "lambda_max": lambda_max, "CI": CI, "CR": CR}



class AHPWeights:
    poi_number: float = 0.539615
    stop_spacing: float = 0.163424
    poi_distance: float = 0.296961

def evaluate_stops_ahp(stops, roads, pois, weights=AHPWeights(), geometria = False):


    # Geometría de la ruta (LineString o MultiLineString)
    route_geometry = unary_union(roads.geometry.values)

    # m-values sobre la ruta
    mvals = stops.geometry.apply(
        lambda p: _snap_point_to_line_and_m(p, route_geometry)[1]
    ).sort_values().to_numpy()

    if len(mvals) > 1:
        spacings = np.diff(mvals)
        sp_full = np.r_[spacings[0],
                        (spacings[:-1] + spacings[1:]) / 2.0,
                        spacings[-1]]
    else:
        sp_full = [0] * len(mvals)

    # Distancia entre paradas
    spacing_score = pd.Series(
        [1.0 if 2000 < d <= 15000 else 0.5 if d <= 25000 else 0.2
         for d in sp_full],
        index=stops.index
    )

    # Número de POIs en un buffer de 5 km
    counts = stops.buffer(5000).apply(lambda b: pois.within(b).sum())
    poi_num_score = counts.apply(
        lambda c: 0.3 if c <= 1 else 0.6 if c <= 3 else 0.8 if c <= 10 else 1.0
    )

    # Distancia media a POIs cercanos
    poi_dist_score = []
    for _, s in stops.iterrows():
        d = pois.distance(s.geometry)
        mask = d <= 5000
        if mask.any():
            avg = d[mask].mean()
            sc = 1.0 if avg <= 2000 else 0.7 if avg <= 3500 else 0.5
        else:
            sc = 0.5
        poi_dist_score.append(sc)
    poi_dist_score = pd.Series(poi_dist_score, index=stops.index)

    # Score total AHP
    total = (weights.poi_number   * poi_num_score +
             weights.stop_spacing * spacing_score +
             weights.poi_distance * poi_dist_score)

    # >>> AQUÍ: devolvemos las paradas + scores <<<
    if geometria:
      result = stops.copy()
      result["poi_num"]     = poi_num_score
      result["spacing"]     = spacing_score
      result["poi_dist"]    = poi_dist_score
      result["total_score"] = total

      result.attrs["overall_mean"] = float(total.mean())
      return result
    else :
      df = pd.DataFrame({"poi_num": poi_num_score,
                         "spacing": spacing_score,
                         "poi_dist": poi_dist_score,
                         "total_score": total})
      df.attrs["overall_mean"] = float(total.mean())
      return df

def optimize_single_corridor(roads, demand_pts, pois, n_new_stops, k_clusters=None,
                             spacing_m=3000, expand_m=1500, shortlist_k=3, min_spacing_m=100):
    candidates = make_candidates_along_route(roads, spacing_m)
    print(f"Candidates: {len(candidates)}")
    candidates2 = expand_candidates(candidates, roads, expand_m)
    print(f"2 Candidates: {len(candidates2)}")
    if k_clusters is None:
        k_clusters = min(20, max(5, int(np.sqrt(len(demand_pts)))))
    centers = kmeans_centers(demand_pts, k=k_clusters)
    sc = wpdm_scores(candidates2, centers, shortlist_k) # Removed 'roads' argument
    new_stops = select_top_n_with_min_spacing(candidates2, sc, n=n_new_stops, min_spacing_m=min_spacing_m)
    print(f"paradas finales: {len(new_stops)}")
    return new_stops, sc