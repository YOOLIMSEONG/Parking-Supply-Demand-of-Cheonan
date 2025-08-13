import numpy as np
import pandas as pd

df_car = pd.read_csv('116_DT_MLTM_5498_20250811152224.csv', encoding='cp949')
df_car.drop(columns='Unnamed: 8', inplace=True)
df_car.info()
df_car['시도명'].unique()
car_districts = df_car['시군구'].unique()
car_districts.shape
df_car['계']

df_parking = pd.read_csv('KC_490_WNTY_PRKLT_2024.csv')
df_parking[df_parking['FLAG_NM'] == '민영']
parking_cheonan = df_parking.loc[df_parking['SIGNGU_NM'].str.contains('천안', na=False)]
parking_cheonan[parking_cheonan['FLAG_NM'] == '민영']
df_parking.info()
df_parking['CTPRVN_NM'].unique()
df_parking['SIGNGU_NM'].unique()
parking_districts = df_parking['SIGNGU_NM'].unique()

df_car = df_car[df_car['시군구'] != '계']
df_car
df_car.groupby('시도명')['계'].sum().dropna()
x1 = df_car.groupby(['시도명','시군구'])['계'].sum().dropna()

df_parking['CTPRVN_NM'] = df_parking['CTPRVN_NM'].replace({'전북특별자치도': '전북'})
df_parking['CTPRVN_NM'] = df_parking['CTPRVN_NM'].replace({'강원특별자치도': '강원'})
df_parking['CTPRVN_NM'] = df_parking['CTPRVN_NM'].replace({'제주특별자치도': '제주'})
df_parking['CTPRVN_NM'] = df_parking['CTPRVN_NM'].replace({'세종특별자치도': '세종'})
df_parking.groupby('CTPRVN_NM')['PARKNG_SPCE_CO'].sum().dropna()
y1 = df_parking.groupby(['CTPRVN_NM','SIGNGU_NM'])['PARKNG_SPCE_CO'].sum().dropna()

df1 = pd.DataFrame({
    '차량 수': x1,
    '주차 가능 대수': y1
}).reset_index()
df1
df1 = df1.rename(columns={'level_0': '시도명', 'level_1': '시군구'})

df1['주차장 보급률'] = df1['주차 가능 대수'] / df1['차량 수'] * 100
df1 = df1.dropna().reset_index()
df1 = df1.drop(columns = 'index')
df1
# df1['지역구'].unique()
# df1[df1['지역구'] == '천안시 동남구']
# df1[df1['지역구'] == '천안시 서북구']
# df1.loc[df1['공영주차장 보급률'].idxmin()]
# df1.loc[df1['공영주차장 보급률'].idxmax()]
# df1.sort_values('공영주차장 보급률')
# df1['공영주차장 보급률'].describe()
# df1[df1['지역구'] == '천안시 서북구'].index
# df1[df1['지역구'] == '천안시 동남구'].index

# 인구 50만 이상 100만 미만 지역구끼리 비교
# 천안
df_cheonan = df1[df1['시군구'].str.contains('천안')].copy()
df_cheonan['지역구'] = '충남 천안시'
df_cheonan = df_cheonan.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_cheonan['주차장 보급률'] = df_cheonan['주차 가능 대수'] / df_cheonan['차량 수'] * 100
df_cheonan
# 경상북도 포항시
df_pohang = df1.loc[df1['시군구'].str.contains('포항')].copy()
df_pohang['지역구'] = '경북 포항시'
df_pohang = df_pohang.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_pohang['주차장 보급률'] = df_pohang['주차 가능 대수'] / df_pohang['차량 수'] * 100
df_pohang
# 경기도 평택시
df_pyeongtaek = df1.loc[df1['시군구'].str.contains('평택')].copy()
df_pyeongtaek['지역구'] = '경기 평택시'
df_pyeongtaek = df_pyeongtaek.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_pyeongtaek['주차장 보급률'] = df_pyeongtaek['주차 가능 대수'] / df_pyeongtaek['차량 수'] * 100
df_pyeongtaek
# 전북특별자치도 전주시
df_jeonju = df1[df1['시군구'].str.contains('전주')].copy()
df_jeonju['지역구'] = '전북 전주시'
df_jeonju = df_jeonju.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_jeonju['주차장 보급률'] = df_jeonju['주차 가능 대수'] / df_jeonju['차량 수'] * 100
df_jeonju
# 경기도 안산시
df_ansan = df1[df1['시군구'].str.contains('안산')].copy()
df_ansan['지역구'] = '경기 안산시'
df_ansan = df_ansan.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_ansan['주차장 보급률'] = df_ansan['주차 가능 대수'] / df_ansan['차량 수'] * 100
df_ansan
# 경기도 안양시
df_anyang = df1[df1['시군구'].str.contains('안양')].copy()
df_anyang['지역구'] = '경기 안양시'
df_anyang = df_anyang.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_anyang['주차장 보급률'] = df_anyang['주차 가능 대수'] / df_anyang['차량 수'] * 100
df_anyang
# 경상남도 김해시
df_gimhae = df1[df1['시군구'].str.contains('김해')].copy()
df_gimhae['지역구'] = '경남 김해시'
df_gimhae = df_gimhae.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_gimhae['주차장 보급률'] = df_gimhae['주차 가능 대수'] / df_gimhae['차량 수'] * 100
df_gimhae
# 경기도 시흥시
df_siheung = df1[df1['시군구'].str.contains('시흥')].copy()
df_siheung['지역구'] = '경기 시흥시'
df_siheung = df_siheung.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_siheung['주차장 보급률'] = df_siheung['주차 가능 대수'] / df_siheung['차량 수'] * 100
df_siheung
# 경기도 파주시
df_paju = df1[df1['시군구'].str.contains('파주')].copy()
df_paju['지역구'] = '경기 파주시'
df_paju = df_paju.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_paju['주차장 보급률'] = df_paju['주차 가능 대수'] / df_paju['차량 수'] * 100
df_paju
# 충청북도 청주시
df_cheongju = df1[df1['시군구'].str.contains('청주')].copy()
df_cheongju['지역구'] = '충북 청주시'
df_cheongju = df_cheongju.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_cheongju['주차장 보급률'] = df_cheongju['주차 가능 대수'] / df_cheongju['차량 수'] * 100
df_cheongju
# 경상남도 창원시
df_changwon = df1[df1['시군구'].str.contains('창원')].copy()
df_changwon['지역구'] = '경남 창원시'
df_changwon = df_changwon.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_changwon['주차장 보급률'] = df_changwon['주차 가능 대수'] / df_changwon['차량 수'] * 100
df_changwon
# 제주특별자치도 제주시
df_jeju = df1[df1['시군구'].str.contains('제주')].copy()
df_jeju['지역구'] = '제주 제주시'
df_jeju = df_jeju.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_jeju['주차장 보급률'] = df_jeju['주차 가능 대수'] / df_jeju['차량 수'] * 100
df_jeju
# 경기도 성남시
df_seongnam = df1[df1['시군구'].str.contains('성남')].copy()
df_seongnam['지역구'] = '경기 성남시'
df_seongnam = df_seongnam.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_seongnam['주차장 보급률'] = df_seongnam['주차 가능 대수'] / df_seongnam['차량 수'] * 100
df_seongnam
# 경기도 부천시
df_bucheon = df1[df1['시군구'].str.contains('부천')].copy()
df_bucheon['지역구'] = '경기 부천시'
df_bucheon = df_bucheon.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_bucheon['주차장 보급률'] = df_bucheon['주차 가능 대수'] / df_bucheon['차량 수'] * 100
df_bucheon
# 경기도 남양주시
df_namyangju = df1[df1['시군구'].str.contains('남양주')].copy()
df_namyangju['지역구'] = '경기 남양주시'
df_namyangju = df_namyangju.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_namyangju['주차장 보급률'] = df_namyangju['주차 가능 대수'] / df_namyangju['차량 수'] * 100
df_namyangju
# 경기도 화성시
df_hwaseong = df1[df1['시군구'].str.contains('화성')].copy()
df_hwaseong['지역구'] = '경기 화성시'
df_hwaseong = df_hwaseong.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_hwaseong['주차장 보급률'] = df_hwaseong['주차 가능 대수'] / df_hwaseong['차량 수'] * 100
df_hwaseong
# 인천광역시 서구
df_seogu = df1[(df1['시군구'] == '서구') & (df1['시도명'] == '인천')].copy()
df_seogu['지역구'] = '인천 서구'
df_seogu = df_seogu.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_seogu['주차장 보급률'] = df_seogu['주차 가능 대수'] / df_seogu['차량 수'] * 100
df_seogu
# 대구광역시 달서구
df_dalseogu = df1[df1['시군구'] == '달서구'].copy()
df_dalseogu['지역구'] = '대구 달서구'
df_dalseogu = df_dalseogu.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_dalseogu['주차장 보급률'] = df_dalseogu['주차 가능 대수'] / df_dalseogu['차량 수'] * 100
df_dalseogu
# 서울특별시 강서구
df_gangseogu = df1[(df1['시군구'] == '강서구') & (df1['시도명'] == '서울')].copy()
df_gangseogu['지역구'] = '서울 강서구'
df_gangseogu = df_gangseogu.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_gangseogu['주차장 보급률'] = df_gangseogu['주차 가능 대수'] / df_gangseogu['차량 수'] * 100
df_gangseogu
# 서울특별시 강남구
df_gangnamgu = df1[df1['시군구'] == '강남구'].copy()
df_gangnamgu['지역구'] = '서울 강남구'
df_gangnamgu = df_gangnamgu.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_gangnamgu['주차장 보급률'] = df_gangnamgu['주차 가능 대수'] / df_gangnamgu['차량 수'] * 100
df_gangnamgu
# 서울특별시 송파구
df_songpa = df1[df1['시군구'].str.contains('송파')].copy()
df_songpa['지역구'] = '서울 송파구'
df_songpa = df_songpa.groupby('지역구')[['차량 수', '주차 가능 대수']].sum().reset_index()
df_songpa['주차장 보급률'] = df_songpa['주차 가능 대수'] / df_songpa['차량 수'] * 100
df_songpa

# 인구수 50만 이상 100만 미만 지역구 정보 취합
df_50_100 = pd.concat(
    [df_cheonan, df_pohang, df_pyeongtaek, df_jeonju, df_ansan, df_anyang,
     df_gimhae, df_siheung, df_paju, df_cheongju, df_changwon, df_jeju,
     df_seongnam, df_bucheon, df_namyangju, df_hwaseong],
     ignore_index=True
).sort_values('주차장 보급률', ascending=False).reset_index()
df_50_100 = df_50_100.drop(columns = 'index')
df_50_100
# 제외: df_gangseogu, df_gangnamgu, df_songpa, df_dalseogu, df_seogu



import pandas as pd
import plotly.express as px

# 원본 보호 + 천안시 여부
df_sorted_50_100 = df_50_100.copy()
df_sorted_50_100['천안시여부'] = df_sorted_50_100['지역구'].str.contains('천안시')

# 보급률 내림차순 정렬
df_sorted_50_100 = df_sorted_50_100.sort_values('주차장 보급률', ascending=False)

# 그래프 높이 자동 조절 (행당 28px 정도)
h = max(500, 28 * len(df_sorted_50_100))

# 막대 그래프 (막대 색상만 하이라이트)
fig = px.bar(
    df_sorted_50_100,
    x='주차장 보급률',
    y='지역구',
    orientation='h',
    text=df_sorted_50_100['주차장 보급률'].map(lambda v: f'{v:.3f}'),
    color='천안시여부',
    color_discrete_map={True: 'red', False: 'lightgray'},
    title='주차장 보급률 (50만~100만 인구 지역구)'
)

# y축 기본 ticklabel 숨기고(글자색 커스터마이즈 위해), 정렬 유지
fig.update_yaxes(
    title='지역구',
    title_standoff=100,                         # ← y축 제목과 라벨 간격 확보
    showticklabels=False,                      # 기본 ticklabel 숨김
    categoryorder='array',
    categoryarray=df_sorted_50_100['지역구'],
    autorange='reversed'
)

# y축 라벨을 annotations로 다시 찍기 (천안시만 빨간 글씨)
for label in df_sorted_50_100['지역구']:
    fig.add_annotation(
        x=0, y=label,
        xref='paper', yref='y',
        text=label,
        showarrow=False,
        font=dict(
            family='Malgun Gothic',
            size=12,
            color='#DC143C' if '천안시' in label else 'black'  # ← 천안시만 Crimson
        ),
        xanchor='right',
        xshift=-10
    )

# 툴팁: 지역구 + 주차장 보급률만 노출 (trace명/기타 숨김)
fig.update_traces(
    hovertemplate='지역구: %{y}<br>주차장 보급률: %{x:.3f}<extra></extra>'
    # 값이 %라면 위 줄의 %{x:.3f} 뒤에 %를 붙이세요: %{x:.3f}%
)

# 레이아웃
fig.update_layout(
    font=dict(family='Malgun Gothic'),
    showlegend=False,
    height=h,
    margin=dict(l=180, r=40, t=60, b=60)
)

fig.show()



# 충청남도 내에서 비교
# 시도명이 충남인 데이터만 뽑아내기
df_chungnam = df1[df1['시도명'] == '충남']
df_chungnam = df_chungnam.sort_values('주차장 보급률', ascending=False)
df_chungnam

# # 충남 데이터 그래프 그리기
# import pandas as pd
# import plotly.graph_objects as go
# 
# font=dict(family='Malgun Gothic')
# 
# # 천안시만 빨간색 하이라이트
# highlight_mask = df_chungnam['시군구'].str.contains('천안시')
# colors = ['red' if flag else 'lightgray' for flag in highlight_mask]
# 
# # 기본 색상 & 강조 색상
# base_color = '#6B7280'   # 회색
# hi_color   = '#1E3A8A'   # 진한 파랑(네이비 계열)
# 
# # 소수점 3자리 문자열 변환 (0.8 → 0.800)
# labels = [f"{val:.3f}" for val in df_chungnam['공영주차장 보급률']]
# 
# # Plotly 그래프
# fig = go.Figure(data=[
#     go.Bar(
#         x=df_chungnam['시군구'],
#         y=df_chungnam['공영주차장 보급률'],
#         marker_color=colors,
#         text=labels,  # 포맷팅된 문자열
#         textposition='outside'
#     )
# ])
# 
# fig.update_layout(
#     title='충남 시군구별 공영주차장 보급률',
#     xaxis_title='시군구',
#     yaxis_title='보급률(%)',
#     font=dict(family='Malgun Gothic'),
#     yaxis=dict(showgrid=True),
#     xaxis=dict(showgrid=False)
# )
# 
# fig.show()



# 충남 데이터 그래프 그리기 (천안시 글자 하이라이트 + y축 여유)
import pandas as pd
import plotly.graph_objects as go

font = dict(family='Malgun Gothic')

# 막대 색 (천안시만 빨강)
highlight_mask = df_chungnam['시군구'].str.contains('천안시')
colors = ['red' if flag else 'lightgray' for flag in highlight_mask]

# 막대 위 숫자(소수점 3자리, 0.8 → 0.800 유지)
labels = [f"{v:.3f}" for v in df_chungnam['주차장 보급률']]

fig = go.Figure([
    go.Bar(
        x=df_chungnam['시군구'],
        y=df_chungnam['주차장 보급률'],
        marker_color=colors,
        text=labels,
        textposition='outside'
    )
])

cats = df_chungnam['시군구'].tolist()
base_color = '#6B7280'
hi_color   = '#DC143C'   # 천안시 라벨 색

# 천안시 외 기본 x축 ticklabel, 천안시는 빈칸 처리
ticktext = [("" if "천안시" in n else n) for n in cats]

fig.update_layout(
    title='충남 시군구별 주차장 보급률',
    font=font,
    yaxis=dict(title='보급률(%)', showgrid=True),
    xaxis=dict(
        tickmode='array',
        tickvals=cats,
        ticktext=ticktext,     # 천안시는 기본 라벨 숨김
        tickangle=45,
        tickfont=dict(color=base_color, size=12),
        showgrid=False
    ),
    margin=dict(l=80, r=40, t=60, b=150)
)

# --- 천안시 두 곳만 빨간 글자로 annotation으로 다시 찍기 ---
for name in cats:
    if '천안시' in name:
        fig.add_annotation(
            x=name, xref='x',
            y=0, yref='paper',
            yanchor='top',
            xanchor='center',
            xshift=28,          # 필요시 16~32 사이로 조절
            yshift=0,
            text=name,
            textangle=45,
            showarrow=False,
            align='center',
            font=dict(family='Malgun Gothic', size=12, color=hi_color)
        )

# --- 숫자 라벨 잘림 방지를 위한 y축 여유 확보 ---
max_val = df_chungnam['주차장 보급률'].max()
fig.update_yaxes(range=[0, max_val * 1.12])   # 여유 12% (상황에 따라 1.10~1.20 조절)

# (보너스) 자동 여백 계산도 함께 켜고 싶다면 아래 한 줄 추가:
# fig.update_yaxes(automargin=True)

fig.show()