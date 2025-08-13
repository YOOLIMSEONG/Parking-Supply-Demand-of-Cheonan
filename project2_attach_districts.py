# attach_districts.py
import os, time
import pandas as pd
import requests
from tqdm import tqdm

# === 0) 설정 ===
IN  = "천안_교차로_카메라매칭_좌표포함.csv"
OUT = "천안_교차로_행정동_정확매핑.csv"

REST_KEY = os.getenv("KAKAO_REST_KEY")
if not REST_KEY:
    raise RuntimeError("환경변수 KAKAO_REST_KEY 가 비어 있습니다.")

# === 1) 데이터 로드 & 좌표 컬럼 찾기 ===
df = pd.read_csv(IN, encoding="utf-8")

def find_col(candidates, keywords):
    for k in keywords:
        for c in candidates:
            if k in str(c).lower():
                return c
    return None

lon_col = find_col(df.columns, ["lon","경도","x","좌표x"])
lat_col = find_col(df.columns, ["lat","위도","y","좌표y"])
if lon_col is None or lat_col is None:
    raise RuntimeError(f"경도/위도 컬럼을 찾지 못했습니다. 현재 컬럼: {list(df.columns)}")

df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")

valid_mask = df["lon"].between(126.0, 128.0) & df["lat"].between(36.0, 37.5)
df_valid = df[valid_mask].copy()  # 인덱스 유지

# === 2) 좌표 캐시 키 ===
def key_ll(lon, lat, nd=6):
    return f"{round(float(lon), nd)}_{round(float(lat), nd)}"

unique_keys = (
    df_valid[["lon","lat"]]
    .dropna()
    .apply(lambda r: key_ll(r["lon"], r["lat"]), axis=1)
    .unique()
)

# === 3) 역지오코딩 ===
def revgeo(lon, lat, session: requests.Session):
    url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
    headers = {"Authorization": f"KakaoAK {REST_KEY}"}
    params  = {"x": float(lon), "y": float(lat)}
    for attempt in range(3):
        try:
            resp = session.get(url, headers=headers, params=params, timeout=5)
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                H = next((d for d in docs if d.get("region_type")=="H"), None)
                B = next((d for d in docs if d.get("region_type")=="B"), None)
                return {
                    "H_si": H and H.get("region_1depth_name"),
                    "H_gu": H and H.get("region_2depth_name"),
                    "H_dong": H and H.get("region_3depth_name"),
                    "H_code": H and H.get("code"),
                    "B_si": B and B.get("region_1depth_name"),
                    "B_gu": B and B.get("region_2depth_name"),
                    "B_dong": B and B.get("region_3depth_name"),
                    "B_code": B and B.get("code"),
                }
            elif resp.status_code in (429, 503):
                time.sleep(1.5 * (attempt + 1))
            else:
                return None
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
    return None

# === 4) API 호출(캐시) ===
cache = {}
with requests.Session() as sess:
    for k in tqdm(unique_keys, desc="Kakao revgeo"):
        try:
            lon, lat = map(float, k.split("_"))
        except Exception:
            cache[k] = None
            continue
        cache[k] = revgeo(lon, lat, sess)
        time.sleep(0.12)

# === 5) 결과 붙이기 — 안전버전(열 길이 불일치 방지) ===
cols = ["H_시도","H_구","H_행정동","H_code","B_시도","B_구","B_법정동","B_code"]

def attach(row):
    lon, lat = row.get("lon"), row.get("lat")
    if pd.isna(lon) or pd.isna(lat):
        return [None]*8
    data = cache.get(key_ll(lon, lat))
    if not data:
        return [None]*8
    return [
        data.get("H_si"), data.get("H_gu"), data.get("H_dong"), data.get("H_code"),
        data.get("B_si"), data.get("B_gu"), data.get("B_dong"), data.get("B_code"),
    ]

# ← 여기만 변경: iterrows로 2D 리스트 만든 뒤 DataFrame 생성
vals = [attach(row) for _, row in df_valid.iterrows()]
tmp  = pd.DataFrame(vals, index=df_valid.index, columns=cols)  # 항상 8열

# 결합
df_valid = pd.concat([df_valid, tmp], axis=1)

# === 6) H 우선, B 보완 ===
def coalesce(a, b):
    return a if pd.notna(a) and a else b

df_valid["구_최종"]       = [coalesce(a,b) for a,b in zip(df_valid["H_구"], df_valid["B_구"])]
df_valid["동_읍면리_최종"] = [coalesce(a,b) for a,b in zip(df_valid["H_행정동"], df_valid["B_법정동"])]

# === 7) 원본과 조인해 저장 ===
extra_cols = cols + ["구_최종","동_읍면리_최종"]
out_df = df.join(df_valid[extra_cols], how="left")
out_df.to_csv(OUT, index=False, encoding="utf-8-sig")
print("완료:", OUT)

print("\n요약")
print("행정구(최종) 분포:\n", out_df["구_최종"].value_counts(dropna=False))
print("\n미확인 좌표 개수:", out_df["구_최종"].isna().sum())