# 천안시 "백화점", "도서관", "영화관",
# "쇼핑몰", "아울렛", "대학교", "종합병원" 크롤링
# 및 공영/민영 주차장 정보 지도에 함께 표시
# 천안시 경계면 정보 가져오기
# -*- coding: utf-8 -*-
"""
Cheonan-only POI Crawler + Polygon Filter + Keyword-colored Map + Parking Layers + CSV
- Adds custom MarkerCluster colors:
  <5: yellow, 5~9: orange, >=10: red, count text = black
"""

import os
import time
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union
import folium
from folium.plugins import MiniMap, MarkerCluster
import webbrowser
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# 설정
# =========================
# Kakao API
KAKAO_REST_KEY = (os.getenv("KAKAO_REST_KEY") or "").strip()
if not KAKAO_REST_KEY:
    raise RuntimeError("환경변수 KAKAO_REST_KEY가 비어있습니다. (Kakao Developers의 REST API 키를 설정하세요)")
HEADERS = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
KAKAO_POI_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_ADDR_URL = "https://dapi.kakao.com/v2/local/search/address.json"

PAGE_SIZE = 15
MAX_PAGES = 45
MAX_FETCHABLE = PAGE_SIZE * MAX_PAGES  # 675
SLEEP_SEC = 0.25
GEOCODE_SLEEP_SEC = 0.2

# 안정적 세션(재시도+백오프)
SESSION = requests.Session()
_retry = Retry(
    total=5, connect=5, read=5,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)
CONNECT_TIMEOUT = 8
READ_TIMEOUT = 25
TIMEOUT_TUPLE = (CONNECT_TIMEOUT, READ_TIMEOUT)

# 경로 (사용자 환경에 맞게 수정)
SHP_PATH = "./project2_cheonan_data/N3A_G0100000/N3A_G0100000.shp"
PUBLIC_PARKING_CSV = "./project2_cheonan_data/천안도시공사_주차장 현황_20250716.csv"   # 주차장명, 주소, 위도, 경도
PRIVATE_PARKING_XLSX = "./project2_cheonan_data/충청남도_천안시_민영주차장정보.xlsx"       # 주차장명, 소재지도로명주소, 소재지지번주소

# 키워드
KEYWORDS = ["백화점", "도서관", "영화관", "쇼핑몰", "아울렛", "대학교", "종합병원"]

# 키워드별 색상
COLOR_CYCLE = ["red","blue","green","purple","orange","darkred","cadetblue","darkblue","darkgreen","lightgray"]
KEYWORD_COLOR = {kw: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, kw in enumerate(KEYWORDS)}
DEFAULT_COLOR = "lightgray"

# 격자
GRID_X = 6
GRID_Y = 4

# 지도
MAP_CENTER_LAT, MAP_CENTER_LON = 36.815, 127.147
MAP_ZOOM = 12
MAP_HTML = "cheonan_keyword_map.html"

# 지오코딩 디스크 캐시
GEOCODE_CACHE_PATH = "geocode_cache.json"
_geocode_cache = {}
if os.path.exists(GEOCODE_CACHE_PATH):
    try:
        with open(GEOCODE_CACHE_PATH, "r", encoding="utf-8") as f:
            _geocode_cache = json.load(f)
    except Exception:
        _geocode_cache = {}

# =========================
# Geo boundary (Cheonan)
# =========================
def load_cheonan_geometry_from_shp(shp_path: str):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # '천안' 자동 탐색
    str_cols = [c for c in gdf.columns if gdf[c].dtype == object]
    mask = pd.Series(False, index=gdf.index)
    for c in str_cols:
        mask |= gdf[c].fillna("").astype(str).str.contains("천안", na=False)

    gdf_cheonan = gdf[mask].copy()
    if len(gdf_cheonan) == 0:
        for term in ["동남구", "서북구", "Cheonan", "Cheonan-si", "Chonan"]:
            tmp = pd.Series(False, index=gdf.index)
            for c in str_cols:
                tmp |= gdf[c].fillna("").astype(str).str.contains(term, na=False)
            if tmp.any():
                gdf_cheonan = pd.concat([gdf_cheonan, gdf[tmp]]).drop_duplicates()
    if len(gdf_cheonan) == 0:
        raise ValueError("Shapefile에서 '천안' 관련 레코드를 찾지 못했습니다.")

    return unary_union(gdf_cheonan.geometry)  # Polygon or MultiPolygon

# =========================
# Kakao API utils
# =========================
def _kakao_get(url, params, headers, max_retries=3):
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = SESSION.get(url, params=params, headers=headers, timeout=TIMEOUT_TUPLE)
        except requests.exceptions.RequestException as e:
            time.sleep(SLEEP_SEC * attempt)
            last = e
            continue

        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(SLEEP_SEC * attempt)
            last = resp
            continue

        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} | url={resp.url}\nbody={resp.text}", response=resp
        )

    if isinstance(last, requests.Response):
        raise requests.HTTPError(
            f"Request failed after retries. last_status={last.status_code}, body={last.text}", response=last
        )
    elif last is not None:
        raise requests.HTTPError(f"Request failed after retries due to network error: {last}")
    else:
        raise requests.HTTPError("Request failed with unknown error")

def whole_region(keyword, minX, minY, maxX, maxY, *, headers=HEADERS, page_size=PAGE_SIZE):
    # 좌표 정렬
    minX, maxX = (minX, maxX) if minX <= maxX else (maxX, minX)
    minY, maxY = (minY, maxY) if minY <= maxY else (maxY, minY)

    page_num = 1
    params = {"query": keyword, "page": page_num, "size": page_size, "rect": f"{minX},{minY},{maxX},{maxY}"}
    resp = _kakao_get(KAKAO_POI_URL, params, headers)
    payload = resp.json()
    total_count = payload.get("meta", {}).get("total_count", 0)

    if total_count > MAX_FETCHABLE:
        docs = []
        midX = (minX + maxX) / 2.0
        midY = (minY + maxY) / 2.0
        docs.extend(whole_region(keyword, minX, minY, midX,  midY,  headers=headers, page_size=page_size))
        docs.extend(whole_region(keyword, midX,  minY, maxX,  midY,  headers=headers, page_size=page_size))
        docs.extend(whole_region(keyword, minX, midY,  midX,  maxY,  headers=headers, page_size=page_size))
        docs.extend(whole_region(keyword, midX,  midY,  maxX,  maxY,  headers=headers, page_size=page_size))
        return docs

    documents = []
    while True:
        cur_payload = payload if page_num == 1 else _kakao_get(KAKAO_POI_URL, {**params, "page": page_num}, headers).json()
        docs = cur_payload.get("documents", [])
        documents.extend(docs)
        is_end = cur_payload.get("meta", {}).get("is_end", True)
        if is_end or page_num >= MAX_PAGES:
            break
        page_num += 1
        time.sleep(SLEEP_SEC)
    return documents

def overlapped_data_in_polygon(keyword, bbox, num_x, num_y, poly, *, headers=HEADERS, page_size=PAGE_SIZE):
    minX, minY, maxX, maxY = bbox
    step_x = (maxX - minX) / float(num_x)
    step_y = (maxY - minY) / float(num_y)

    results = []
    for i in range(num_x):
        for j in range(num_y):
            cell_minX = minX + i * step_x
            cell_maxX = cell_minX + step_x
            cell_minY = minY + j * step_y
            cell_maxY = cell_minY + step_y
            cell_geom = box(cell_minX, cell_minY, cell_maxX, cell_maxY)
            if not poly.intersects(cell_geom):
                continue
            docs = whole_region(keyword, cell_minX, cell_minY, cell_maxX, cell_maxY,
                                headers=headers, page_size=page_size)
            results.extend(docs)
            time.sleep(SLEEP_SEC)
    return results

# =========================
# Geocoding & Parking loaders
# =========================
def geocode_address(addr: str, headers=HEADERS, per_addr_retries=3):
    """주소 -> (lon, lat). 재시도 + 디스크 캐시 사용."""
    if not addr or not str(addr).strip():
        return (np.nan, np.nan)
    key = str(addr).strip()

    # 메모리 캐시
    if key in _geocode_cache:
        lon, lat = _geocode_cache[key]
        return (float(lon), float(lat)) if (lon is not None and lat is not None) else (np.nan, np.nan)

    params = {"query": key}
    last_exc = None
    for attempt in range(1, per_addr_retries + 1):
        try:
            resp = _kakao_get(KAKAO_ADDR_URL, params, headers)
            data = resp.json()
            docs = data.get("documents", [])
            if not docs:
                _geocode_cache[key] = (None, None)
                break
            x = docs[0].get("x"); y = docs[0].get("y")
            lon = float(x); lat = float(y)
            _geocode_cache[key] = (lon, lat)
            # 디스크 캐시 flush
            try:
                with open(GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(_geocode_cache, f, ensure_ascii=False)
            except Exception:
                pass
            time.sleep(GEOCODE_SLEEP_SEC)
            return (lon, lat)
        except (requests.exceptions.RequestException, requests.HTTPError) as e:
            last_exc = e
            time.sleep(SLEEP_SEC * attempt)
            continue

    _geocode_cache[key] = (None, None)
    if last_exc:
        print(f"[WARN] Geocode failed after retries: {key} | {last_exc}")
    return (np.nan, np.nan)

def load_public_parking(csv_path: str):
    try:
        dfp = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        dfp = pd.read_csv(csv_path, encoding="cp949")

    rename = {}
    if "주차장명" in dfp.columns: rename["주차장명"] = "name"
    if "주소" in dfp.columns: rename["주소"] = "address"
    if "위도" in dfp.columns: rename["위도"] = "lat"
    if "경도" in dfp.columns: rename["경도"] = "lon"
    dfp = dfp.rename(columns=rename)

    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    dfp["road_address"] = dfp["address"].fillna("")
    dfp["jibun_address"] = ""
    dfp["category"] = "공영주차장"
    dfp["source"] = "천안도시공사"
    dfp["id"] = "public_" + dfp.index.astype(str)

    return dfp[["id","name","lat","lon","road_address","jibun_address","category","source"]]

def load_private_parking(xlsx_path: str):
    dfm = pd.read_excel(xlsx_path)
    rename = {}
    if "주차장명" in dfm.columns: rename["주차장명"] = "name"
    if "소재지도로명주소" in dfm.columns: rename["소재지도로명주소"] = "road_address"
    if "소재지지번주소" in dfm.columns: rename["소재지지번주소"] = "jibun_address"
    dfm = dfm.rename(columns=rename)

    lons, lats = [], []
    for _, row in dfm.iterrows():
        cand1 = str(row.get("road_address") or "").strip()
        cand2 = str(row.get("jibun_address") or "").strip()
        query = cand1 if cand1 else cand2
        lon, lat = geocode_address(query)
        lons.append(lon); lats.append(lat)

    dfm["lon"] = lons
    dfm["lat"] = lats
    dfm["category"] = "민영주차장"
    dfm["source"] = "천안시/민영"
    dfm["id"] = "private_" + dfm.index.astype(str)

    return dfm[["id","name","lat","lon","road_address","jibun_address","category","source"]]

# =========================
# 클러스터 아이콘(JS) 커스터마이저
# =========================
def make_cluster(thresholds=(5, 10)):
    """
    MarkerCluster를 그라데이션 스타일로 커스터마이징합니다.
    - <t1: 노란색, [t1,t2): 주황색, >=t2: 빨간색 (그라데이션)
    - 숫자 색상: 검정
    - 테두리: 은은한 1px
    """
    t1, t2 = thresholds
    js = f"""
    function(cluster) {{
        var count = cluster.getChildCount();

        // 기본값: 노란색 그라데이션
        var grad = 'radial-gradient(circle at 30% 30%, #fff3b0 0%, #ffe066 55%, #ffc107 100%)';

        // 5~9: 주황색, 10 이상: 빨간색
        if (count >= {t2}) {{
            grad = 'radial-gradient(circle at 30% 30%, #ffb3b3 0%, #ff6b6b 55%, #e03131 100%)';
        }} else if (count >= {t1}) {{
            grad = 'radial-gradient(circle at 30% 30%, #ffd6a5 0%, #ff922b 55%, #f76707 100%)';
        }}

        var html = ''
            + '<div style="'
            + 'background:' + grad + ';'
            + 'border:1px solid rgba(0,0,0,0.25);'
            + 'border-radius:50%;'
            + 'width:40px;height:40px;'
            + 'display:flex;align-items:center;justify-content:center;'
            + 'box-shadow:0 0 0 2px rgba(255,255,255,0.6) inset;'
            + '">'
            + '<span style="color:black;font-weight:700;">' + count + '</span>'
            + '</div>';

        return new L.DivIcon({{
            html: html,
            className: 'marker-cluster-custom',
            iconSize: new L.Point(40, 40)
        }});
    }}
    """
    return MarkerCluster(icon_create_function=js)

# =========================
# 지도 (키워드별 레이어 + 주차장 레이어)
# =========================
def make_map_keyword_layers(df, center_lat=MAP_CENTER_LAT, center_lon=MAP_CENTER_LON, zoom_start=MAP_ZOOM):
    df = df.copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    if len(df):
        center_lat = float(df["lat"].mean())
        center_lon = float(df["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    m.add_child(MiniMap())

    ICON_BY_KEYWORD = {
        "백화점": ("shopping-cart", "fa"),
        "도서관": ("book", "fa"),
        "영화관": ("film", "fa"),
        "쇼핑몰": ("shopping-bag", "fa"),
        "아울렛": ("tags", "fa"),
        "대학교": ("university", "fa"),
        "종합병원": ("hospital-o", "fa"),
    }

    layers = {}
    keywords_sorted = sorted([k for k in df["primary_keyword"].dropna().unique()] + ["(미분류)"])
    for kw in keywords_sorted:
        fg = folium.FeatureGroup(name=f"[시설] {kw}")
        # 커스텀 클러스터 사용
        cluster = make_cluster().add_to(fg)
        layers[kw] = {"fg": fg, "cluster": cluster}
        m.add_child(fg)

    for _, row in df.iterrows():
        name = str(row.get("name", ""))
        road_addr = str(row.get("road_address", ""))
        jibun_addr = str(row.get("jibun_address", ""))
        phone = str(row.get("phone", ""))
        url = str(row.get("url", ""))
        all_kws = row.get("keywords", "")
        primary_kw = row.get("primary_keyword", "(미분류)")
        color = KEYWORD_COLOR.get(primary_kw, DEFAULT_COLOR)
        icon_name, icon_prefix = ICON_BY_KEYWORD.get(primary_kw, ("info-sign", "glyphicon"))

        # 툴팁(커서 올릴 때)
        tooltip_text = f"{name}\n도로명: {road_addr}\n지번: {jibun_addr}"
        tooltip = folium.Tooltip(tooltip_text, sticky=True)

        # 팝업(클릭 시)
        popup_html = f"""
        <div style="font-size:14px;line-height:1.4;">
            <b>{name}</b><br>
            <b>키워드</b> : {all_kws}<br>
            <b>도로명</b> : {road_addr}<br>
            <b>지번</b> : {jibun_addr}<br>
            <b>전화</b> : {phone}<br>
            <a href="{url}" target="_blank" rel="noopener">카카오 장소 페이지</a>
        </div>
        """

        marker = folium.Marker(
            [row["lat"], row["lon"]],
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=350),
            icon=folium.Icon(color=color, icon=icon_name, prefix=icon_prefix),
        )
        layer_key = primary_kw if primary_kw in layers else "(미분류)"
        marker.add_to(layers[layer_key]["cluster"])

    # LayerControl은 메인에서 마지막에 한 번만 추가
    return m

def add_parking_layers_to_map(m, df_public, df_private):
    # 공영
    fg_pub = folium.FeatureGroup(name="[주차장] 공영")
    cluster_pub = make_cluster().add_to(fg_pub)
    for _, r in df_public.iterrows():
        if pd.isna(r["lat"]) or pd.isna(r["lon"]):
            continue
        name = str(r.get("name",""))
        road_addr = str(r.get("road_address",""))
        jibun_addr = str(r.get("jibun_address",""))
        tooltip = folium.Tooltip(f"{name}\n도로명: {road_addr}\n지번: {jibun_addr}", sticky=True)
        html = f"""
        <div style="font-size:14px;line-height:1.4;">
            <b>{name}</b><br>
            <b>유형</b> : 공영주차장<br>
            <b>도로명</b> : {road_addr}<br>
            <b>지번</b> : {jibun_addr}<br>
            <b>출처</b> : {str(r.get('source',''))}
        </div>
        """
        folium.Marker(
            [float(r["lat"]), float(r["lon"])],
            tooltip=tooltip,
            popup=folium.Popup(html, max_width=350),
            icon=folium.Icon(color="black", icon="car", prefix="fa")
        ).add_to(cluster_pub)
    m.add_child(fg_pub)

    # 민영
    fg_pri = folium.FeatureGroup(name="[주차장] 민영")
    cluster_pri = make_cluster().add_to(fg_pri)
    for _, r in df_private.iterrows():
        if pd.isna(r["lat"]) or pd.isna(r["lon"]):
            continue
        name = str(r.get("name",""))
        road_addr = str(r.get("road_address",""))
        jibun_addr = str(r.get("jibun_address",""))
        tooltip = folium.Tooltip(f"{name}\n도로명: {road_addr}\n지번: {jibun_addr}", sticky=True)
        html = f"""
        <div style="font-size:14px;line-height:1.4;">
            <b>{name}</b><br>
            <b>유형</b> : 민영주차장<br>
            <b>도로명</b> : {road_addr}<br>
            <b>지번</b> : {jibun_addr}<br>
            <b>출처</b> : {str(r.get('source',''))}
        </div>
        """
        folium.Marker(
            [float(r["lat"]), float(r["lon"])],
            tooltip=tooltip,
            popup=folium.Popup(html, max_width=350),
            icon=folium.Icon(color="gray", icon="car", prefix="fa")
        ).add_to(cluster_pri)
    m.add_child(fg_pri)

    return m

# =========================
# Helpers
# =========================
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# =========================
# 메인
# =========================
if __name__ == "__main__":
    # 1) 천안 경계 & bbox
    cheonan_geom = load_cheonan_geometry_from_shp(SHP_PATH)
    minX, minY, maxX, maxY = cheonan_geom.bounds
    print(f"[INFO] Cheonan bbox: ({minX:.6f}, {minY:.6f}) ~ ({maxX:.6f}, {maxY:.6f})")

    # 2) 키워드별 수집 (경계 교차 타일만), id 중복 제거 + 키워드 누적
    by_id = {}  # id -> doc + _kws(list)
    for kw in KEYWORDS:
        print(f"[INFO] Collecting keyword: {kw}")
        raw_docs = overlapped_data_in_polygon(
            kw, (minX, minY, maxX, maxY), GRID_X, GRID_Y, cheonan_geom,
            headers=HEADERS, page_size=PAGE_SIZE
        )
        for d in raw_docs:
            # '대학교' 수집 시 '와플대학' 제외
            if kw == "대학교" and "와플대학" in str(d.get("place_name", "")):
                continue
            pid = d.get("id")
            if not pid:
                continue
            if pid not in by_id:
                by_id[pid] = {**d, "_kws": [kw]}
            else:
                kws = by_id[pid].get("_kws", [])
                if kw not in kws:
                    kws.append(kw)
                by_id[pid]["_kws"] = kws

    results = list(by_id.values())

    # 3) DataFrame 변환
    df = pd.DataFrame({
        "id":   [p.get("id", "") for p in results],
        "name": [p.get("place_name", "") for p in results],
        "lon":  [_safe_float(p.get("x")) for p in results],  # 경도
        "lat":  [_safe_float(p.get("y")) for p in results],  # 위도
        "road_address": [p.get("road_address_name", "") for p in results],
        "jibun_address": [p.get("address_name", "") for p in results],
        "url":  [p.get("place_url", "") for p in results],
        "category": [p.get("category_name", "") for p in results],
        "phone": [p.get("phone", "") for p in results],
        "keywords": [";".join(p.get("_kws", [])) for p in results],
        "primary_keyword": [p.get("_kws", [None])[0] if p.get("_kws") else None for p in results],
    })
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # 4) 천안 경계 내부로 정확 필터링 + 와플대학 안전망 제외
    mask_in = df.apply(lambda r: cheonan_geom.contains(Point(float(r["lon"]), float(r["lat"]))), axis=1)
    df_in = df[mask_in].reset_index(drop=True)
    df_in = df_in[~(df_in["name"].str.contains("와플대학", na=False) &
                    df_in["keywords"].str.contains("대학교", na=False))].reset_index(drop=True)

    print(f"[INFO] total_collected (dedup by id): {len(df)}")
    print(f"[INFO] inside Cheonan polygon:       {len(df_in)}")

    # 5) CSV 저장 (POI 전체 + 키워드별)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    all_csv = f"cheonan_POI_all_{ts}.csv"
    df_in.to_csv(all_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved CSV (all): {all_csv}")
    for kw, sub in df_in.groupby("primary_keyword"):
        if not kw:
            continue
        fn = f"cheonan_POI_{kw}_{ts}.csv"
        sub.to_csv(fn, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved CSV ({kw}): {fn}")

    # 6) 주차장 로드 + 천안 내부 필터 + CSV 저장
    df_pub = load_public_parking(PUBLIC_PARKING_CSV)
    df_pri = load_private_parking(PRIVATE_PARKING_XLSX)

    def _inside(poly, lon, lat):
        try:
            return poly.contains(Point(float(lon), float(lat)))
        except Exception:
            return False

    if len(df_pub):
        df_pub = df_pub[df_pub.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)
    if len(df_pri):
        df_pri = df_pri[df_pri.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

    pub_csv = f"cheonan_public_parking_{ts}.csv"
    pri_csv = f"cheonan_private_parking_{ts}.csv"
    df_pub.to_csv(pub_csv, index=False, encoding="utf-8-sig")
    df_pri.to_csv(pri_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved CSV (public): {pub_csv}")
    print(f"[INFO] Saved CSV (private): {pri_csv}")

    # 7) 지도 생성(키워드 레이어) + 주차장 레이어 추가 → LayerControl(마지막에 1회) → 저장 & 자동열기
    m = make_map_keyword_layers(df_in, center_lat=MAP_CENTER_LAT, center_lon=MAP_CENTER_LON, zoom_start=MAP_ZOOM)
    m = add_parking_layers_to_map(m, df_pub, df_pri)
    folium.LayerControl(collapsed=False).add_to(m)  # 모든 레이어 추가 후 딱 1회
    m.save(MAP_HTML)
    print(f"[INFO] Saved map to {MAP_HTML}")
    webbrowser.open('file://' + os.path.realpath(MAP_HTML))
