import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import os
import json

st.set_page_config(page_title="ハイブリッド株式スクリーナー", layout="wide")
st.title("ハイブリッド株式スクリーナー ＆ ファクター分析")

# ==========================================
# 🌟 ユーザー設定の保存・読み込み機能
# ==========================================
SETTINGS_FILE = "user_settings.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "favorites": ["4661"], 
        "scr_roe": 0.0, "scr_per": 100.0, "scr_div": 0.0,
        "scr_pbr": 10.0, "scr_growth": -50.0, "scr_equity": 0.0
    }

def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

if 'settings' not in st.session_state:
    st.session_state['settings'] = load_settings()
    st.session_state['favorites'] = st.session_state['settings'].get("favorites", ["4661"])
    st.session_state['scr_roe'] = st.session_state['settings'].get("scr_roe", 0.0)
    st.session_state['scr_per'] = st.session_state['settings'].get("scr_per", 100.0)
    st.session_state['scr_div'] = st.session_state['settings'].get("scr_div", 0.0)
    st.session_state['scr_pbr'] = st.session_state['settings'].get("scr_pbr", 10.0)
    st.session_state['scr_growth'] = st.session_state['settings'].get("scr_growth", -50.0)
    st.session_state['scr_equity'] = st.session_state['settings'].get("scr_equity", 0.0)

def add_favorite(code):
    if code not in st.session_state['favorites']:
        st.session_state['favorites'].append(code)
        st.session_state['settings']['favorites'] = st.session_state['favorites']
        save_settings(st.session_state['settings'])

def remove_favorite(code):
    if code in st.session_state['favorites']:
        st.session_state['favorites'].remove(code)
        st.session_state['settings']['favorites'] = st.session_state['favorites']
        save_settings(st.session_state['settings'])

# ==========================================
# ユーティリティ関数
# ==========================================
@st.cache_data
def load_and_clean_master_data(filepath="master_database.csv"):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, encoding="cp932", on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines='skip')

    if '前日比(%)' in df.columns:
        if df['前日比(%)'].dtype == 'object':
            extracted = df['前日比(%)'].astype(str).str.extract(r'\(([-+]?[0-9]*\.?[0-9]+)%\)')
            df['前日比(%) (数値)'] = pd.to_numeric(extracted[0], errors='coerce')
        else:
            df['前日比(%) (数値)'] = df['前日比(%)']
    
    for col in df.columns:
        if df[col].dtype == 'object':
            if col not in ['取得日', 'コード', '銘柄名', '市場', '財務', '前日比(%)']:
                cleaned = df[col].astype(str).str.strip().str.replace(',', '', regex=True)
                df[col] = pd.to_numeric(cleaned, errors='coerce')
    
    # 取得日の時間を消して「YYYY-MM-DD」形式のみにする
    if '取得日' in df.columns:
        df['取得日'] = pd.to_datetime(df['取得日'], errors='coerce').dt.strftime('%Y-%m-%d')

    return df

def get_ticker_symbol(code):
    return f"{str(code).strip()}.T"

def format_metric(val, suffix=""):
    if pd.isna(val) or str(val).lower() == "nan":
        return "ー"
    if isinstance(val, (int, float)):
        return f"{val:g}{suffix}"
    return f"{val}{suffix}"

# ==========================================
# UI構築とメイン処理
# ==========================================
st.sidebar.header("分析モード")
mode = st.sidebar.radio(
    "選択してください",
    [
        "🔍 個別銘柄の検索・チャート分析",
        "⭐ スクリーニング ＆ お気に入り",
        "📈 現在の要因分析 (多変量解析)", 
        "⏱️ 過去データでの答え合わせ (バックテスト)"
    ]
)
st.sidebar.markdown("---")

with st.spinner("マスターデータを自動読み込み中..."):
    master_df = load_and_clean_master_data()

if master_df is not None:
    st.sidebar.success(f"✅ マスターデータを自動読み込みしました\n(総データ数: {len(master_df)}件)")
    
    exclude_cols = ['コード', '前日比(%)']
    numeric_cols = [c for c in master_df.select_dtypes(include=['float64', 'int64']).columns if c not in exclude_cols]
    
    if '取得日' in master_df.columns:
        latest_df = master_df.sort_values('取得日').groupby('コード').tail(1)
    else:
        latest_df = master_df.copy()

    # --- モード1: 検索・チャート ---
    if mode == "🔍 個別銘柄の検索・チャート分析":
        st.subheader("個別銘柄の検索と推移の確認")
        
        latest_df['検索用ラベル'] = latest_df['コード'].astype(str) + " : " + latest_df['銘柄名'].astype(str)
        selected_label = st.selectbox("企業名やコードを入力して検索してください", latest_df['検索用ラベル'].unique())
        
        if selected_label:
            selected_code = selected_label.split(" : ")[0]
            stock_info_latest = latest_df[latest_df['コード'].astype(str) == selected_code].iloc[0]
            
            col_title, col_btn = st.columns([4, 1])
            with col_title:
                st.markdown(f"### {stock_info_latest['銘柄名']} ({selected_code}) の最新データ")
            with col_btn:
                if selected_code in st.session_state['favorites']:
                    if st.button("⭐ お気に入り解除"):
                        remove_favorite(selected_code)
                        st.rerun()
                else:
                    if st.button("☆ お気に入りに登録"):
                        add_favorite(selected_code)
                        st.rerun()
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("株価(CSV時点)", format_metric(stock_info_latest.get('現在値')))
            col2.metric("ROE", format_metric(stock_info_latest.get('ROE(自己資本利益率)(%)'), "%"))
            col3.metric("PER", format_metric(stock_info_latest.get('PER(株価収益率)(倍)'), "倍"))
            col4.metric("PBR", format_metric(stock_info_latest.get('PBR(株価純資産倍率)(倍)'), "倍"))
            col5.metric("配当利回り", format_metric(stock_info_latest.get('配当利回り(%)'), "%"))
            col6.metric("EPS(予)", format_metric(stock_info_latest.get('EPS(予)(一株あたり当期利益)(円)'), "円"))
            
            st.markdown("#### 詳細指標")
            exclude_from_details = [
                '取得日', 'コード', '銘柄名', '市場', '財務', '検索用ラベル', '前日比(%)', '前日比(%) (数値)',
                '現在値', 'ROE(自己資本利益率)(%)', 'PER(株価収益率)(倍)', 
                'PBR(株価純資産倍率)(倍)', '配当利回り(%)', 'EPS(予)(一株あたり当期利益)(円)'
            ]
            display_cols = [c for c in latest_df.columns if c not in exclude_from_details]
            
            d_col1, d_col2 = st.columns(2)
            for i, col_name in enumerate(display_cols):
                val = stock_info_latest.get(col_name)
                suffix = "%" if "(%)" in col_name else "倍" if "(倍)" in col_name else "円" if "(円)" in col_name else ""
                formatted_val = format_metric(val, suffix)
                if i % 2 == 0:
                    d_col1.markdown(f"**{col_name}**: {formatted_val}")
                else:
                    d_col2.markdown(f"**{col_name}**: {formatted_val}")
            
            st.divider()
            
            st.write(f"**【自動取得】過去1年間の株価推移 ＆ ボリンジャーバンド(±2σ)**")
            with st.spinner("株価の推移データを取得中..."):
                try:
                    ticker_symbol = get_ticker_symbol(selected_code)
                    history = yf.download(ticker_symbol, period="1y")
                    if not history.empty:
                        history['MA25'] = history['Close'].rolling(window=25).mean()
                        history['STD25'] = history['Close'].rolling(window=25).std()
                        history['Upper2'] = history['MA25'] + (history['STD25'] * 2)
                        history['Lower2'] = history['MA25'] - (history['STD25'] * 2)

                        fig = go.Figure(data=[go.Candlestick(
                            x=history.index, open=history['Open'], high=history['High'],
                            low=history['Low'], close=history['Close'], name="株価"
                        )])
                        
                        fig.add_trace(go.Scatter(x=history.index, y=history['Upper2'], mode='lines', name='+2σ (買われすぎ目安)', line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dash')))
                        fig.add_trace(go.Scatter(x=history.index, y=history['Lower2'], mode='lines', name='-2σ (売られすぎ目安)', line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(200, 200, 200, 0.1)'))
                        fig.add_trace(go.Scatter(x=history.index, y=history['MA25'], mode='lines', name='25日移動平均', line=dict(color='blue', width=1.5)))
                        
                        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error("チャート取得失敗")

            # 🌟 今回追加・修正したCSVの推移グラフ部分
            st.divider()
            st.write(f"**【CSVデータ】{stock_info_latest['銘柄名']} のファンダメンタル指標推移**")
            
            stock_history_df = master_df[master_df['コード'].astype(str) == selected_code].sort_values('取得日')
            
            if len(stock_history_df) > 1 and '取得日' in stock_history_df.columns:
                plot_cols = [c for c in numeric_cols if c not in ['コード', '現在値', '前日比(%) (数値)']]
                default_idx = plot_cols.index('ROE(自己資本利益率)(%)') if 'ROE(自己資本利益率)(%)' in plot_cols else 0
                
                selected_metric = st.selectbox("推移を確認したい指標を選択してください:", plot_cols, index=default_idx)
                
                if selected_metric:
                    fig_metric = px.line(
                        stock_history_df.dropna(subset=[selected_metric]), 
                        x='取得日', 
                        y=selected_metric, 
                        markers=True,
                        title=f"{stock_info_latest['銘柄名']} の {selected_metric} の推移"
                    )
                    fig_metric.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_metric, use_container_width=True)
            else:
                st.info("※この銘柄の過去の記録（複数日分のデータ）がマスターCSV内に存在しないため、推移グラフは表示されません。日々データを追記していくことでグラフ化されるようになります。")


    # --- モード2: スクリーニング ＆ お気に入り ---
    elif mode == "⭐ スクリーニング ＆ お気に入り":
        tab1, tab2 = st.tabs(["🎯 条件スクリーニング", "⭐ お気に入りリスト"])
        
        with tab1:
            st.subheader("条件を設定して銘柄を絞り込む")
            
            st.write("▼ プリセット条件で一発セット")
            col_p1, col_p2, col_p3 = st.columns(3)
            
            if col_p1.button("📈 成長＆安全性重視（画像1）"):
                st.session_state['scr_roe'] = 10.0
                st.session_state['scr_growth'] = 10.0
                st.session_state['scr_equity'] = 40.0
                st.session_state['scr_per'] = 100.0
                st.session_state['scr_pbr'] = 10.0
                st.session_state['scr_div'] = 0.0
                
            if col_p2.button("💎 厳格バリュー基準（画像2）"):
                st.session_state['scr_roe'] = 10.0
                st.session_state['scr_per'] = 15.0
                st.session_state['scr_pbr'] = 2.0
                st.session_state['scr_growth'] = -50.0
                st.session_state['scr_equity'] = 0.0
                st.session_state['scr_div'] = 0.0
                
            if col_p3.button("🔄 すべてリセット"):
                st.session_state['scr_roe'] = 0.0
                st.session_state['scr_per'] = 100.0
                st.session_state['scr_pbr'] = 10.0
                st.session_state['scr_growth'] = -50.0
                st.session_state['scr_equity'] = 0.0
                st.session_state['scr_div'] = 0.0

            st.divider()

            st.write("▼ 個別条件の微調整（スライダーを動かすか、右側の枠に直接数値を入力できます）")
            
            def dual_input(label, key, min_val, max_val):
                col_slider, col_num = st.columns([3, 1])
                with col_slider:
                    val_slider = st.slider(label, min_val, max_val, float(st.session_state.get(key, 0.0)), step=0.1, key=f"{key}_slider")
                with col_num:
                    val_num = st.number_input("直接入力", min_value=min_val, max_value=max_val, value=val_slider, step=0.1, key=f"{key}_num", label_visibility="collapsed")
                st.session_state[key] = val_num
                return val_num

            current_roe = dual_input("ROEの最低ライン (%)", 'scr_roe', 0.0, 50.0)
            current_per = dual_input("PERの上限 (倍)", 'scr_per', 0.0, 100.0)
            current_pbr = dual_input("PBRの上限 (倍)", 'scr_pbr', 0.0, 10.0)
            current_div = dual_input("配当利回りの最低ライン (%)", 'scr_div', 0.0, 10.0)
            current_growth = dual_input("3年平均売上成長率の最低ライン (%)", 'scr_growth', -50.0, 50.0)
            current_equity = dual_input("自己資本比率の最低ライン (%)", 'scr_equity', 0.0, 100.0)
            
            st.write("")
            if st.button("💾 現在の条件をデフォルトとして保存"):
                save_settings(st.session_state['settings'])
                st.success("条件を保存しました！次回起動時もこの条件がセットされます。")
            
            for col in ['ROE(自己資本利益率)(%)', 'PER(株価収益率)(倍)', 'PBR(株価純資産倍率)(倍)', '配当利回り(%)', '過去3年平均売上高成長率(予)(%)', '自己資本比率(%)']:
                if col in latest_df.columns:
                    latest_df[col] = pd.to_numeric(latest_df[col], errors='coerce')

            filtered_df = latest_df[
                (latest_df['ROE(自己資本利益率)(%)'].fillna(0) >= current_roe) &
                (latest_df['PER(株価収益率)(倍)'].fillna(100) <= current_per) &
                (latest_df['PBR(株価純資産倍率)(倍)'].fillna(10) <= current_pbr) &
                (latest_df['配当利回り(%)'].fillna(0) >= current_div)
            ]

            if '過去3年平均売上高成長率(予)(%)' in latest_df.columns:
                filtered_df = filtered_df[filtered_df['過去3年平均売上高成長率(予)(%)'].fillna(-100) >= current_growth]
            if '自己資本比率(%)' in latest_df.columns:
                filtered_df = filtered_df[filtered_df['自己資本比率(%)'].fillna(0) >= current_equity]
            
            st.success(f"条件に合致する銘柄: {len(filtered_df)}件")
            
            if len(filtered_df) > 0 and st.button("🚀 絞り込んだ銘柄の最新株価を一括取得"):
                with st.spinner(f"{len(filtered_df)}件の株価を取得中..."):
                    tickers = [get_ticker_symbol(code) for code in filtered_df['コード'].unique()]
                    try:
                        data = yf.download(tickers, period="1d", group_by="ticker", threads=False)
                        current_prices = {}
                        for ticker in tickers:
                            try:
                                if len(tickers) == 1: current_prices[ticker] = data['Close'].iloc[-1]
                                else: current_prices[ticker] = data[ticker]['Close'].iloc[-1]
                            except: pass
                        filtered_df['最新株価'] = filtered_df['コード'].apply(lambda x: current_prices.get(get_ticker_symbol(x)))
                        
                        display_cols = ['コード', '銘柄名', 'ROE(自己資本利益率)(%)', 'PER(株価収益率)(倍)', 'PBR(株価純資産倍率)(倍)']
                        if '過去3年平均売上高成長率(予)(%)' in filtered_df.columns: display_cols.append('過去3年平均売上高成長率(予)(%)')
                        if '自己資本比率(%)' in filtered_df.columns: display_cols.append('自己資本比率(%)')
                        display_cols.append('最新株価')
                        st.dataframe(filtered_df[display_cols])
                    except:
                        st.error("取得失敗")
            else:
                display_cols = ['コード', '銘柄名', 'ROE(自己資本利益率)(%)', 'PER(株価収益率)(倍)', 'PBR(株価純資産倍率)(倍)']
                if '過去3年平均売上高成長率(予)(%)' in filtered_df.columns: display_cols.append('過去3年平均売上高成長率(予)(%)')
                if '自己資本比率(%)' in filtered_df.columns: display_cols.append('自己資本比率(%)')
                display_cols.append('現在値')
                st.dataframe(filtered_df[display_cols])

        with tab2:
            st.subheader("お気に入り銘柄の動向チェック")
            fav_codes = st.session_state['favorites']
            
            if not fav_codes:
                st.info("お気に入りに登録されている銘柄はありません。「個別銘柄の検索」から追加してください。")
            else:
                fav_df = latest_df[latest_df['コード'].astype(str).isin(fav_codes)].copy()
                
                hidden_cols = ['検索用ラベル', '前日比(%) (数値)']
                
                if st.button("🚀 お気に入り銘柄の最新株価を取得"):
                    with st.spinner("取得中..."):
                        tickers = [get_ticker_symbol(code) for code in fav_df['コード'].unique()]
                        try:
                            data = yf.download(tickers, period="1d", group_by="ticker", threads=False)
                            current_prices = {}
                            for ticker in tickers:
                                try:
                                    if len(tickers) == 1: current_prices[ticker] = data['Close'].iloc[-1]
                                    else: current_prices[ticker] = data[ticker]['Close'].iloc[-1]
                                except: pass
                            fav_df['最新株価(Yahoo)'] = fav_df['コード'].apply(lambda x: current_prices.get(get_ticker_symbol(x)))
                            
                            past_price_col = '現在値' if '現在値' in fav_df.columns else '当時の株価'
                            if past_price_col in fav_df.columns:
                                fav_df['変動率(%)'] = (fav_df['最新株価(Yahoo)'] - fav_df[past_price_col]) / fav_df[past_price_col] * 100
                                
                            front_cols = ['取得日', 'コード', '銘柄名', past_price_col, '最新株価(Yahoo)', '変動率(%)']
                            other_cols = [c for c in fav_df.columns if c not in front_cols and c not in hidden_cols]
                            
                            st.dataframe(fav_df[front_cols + other_cols])
                        except:
                            st.error("取得失敗")
                else:
                    front_cols = ['取得日', 'コード', '銘柄名', '現在値'] if '現在値' in fav_df.columns else ['取得日', 'コード', '銘柄名']
                    other_cols = [c for c in fav_df.columns if c not in front_cols and c not in hidden_cols]
                    
                    st.dataframe(fav_df[front_cols + other_cols])

    # --- モード3: 現在の要因分析 ---
    elif mode == "📈 現在の要因分析 (多変量解析)":
        st.subheader("現在のデータに潜む要因の分析")
        
        target_mode = st.radio(
            "分析のアプローチ（目的変数）を選択してください:", 
            ["📊 CSV内の指標（前日比やROEなど）を目的変数にする", 
             "🚀 中長期的な「株価上昇率」を自動取得して目的変数にする"]
        )
        st.divider()

        if target_mode == "📊 CSV内の指標（前日比やROEなど）を目的変数にする":
            if len(numeric_cols) > 1:
                col_target, col_features = st.columns([1, 2])
                with col_target:
                    default_idx = numeric_cols.index('前日比(%) (数値)') if '前日比(%) (数値)' in numeric_cols else 0
                    target_col = st.selectbox("目的変数を選択:", numeric_cols, index=default_idx)
                with col_features:
                    available_features = [c for c in numeric_cols if c != target_col]
                    default_feats = [f for f in available_features if "ROE" in f or "PER" in f or "自己資本比率" in f][:3]
                    feature_cols = st.multiselect("説明変数を選択:", available_features, default=default_feats)

                if feature_cols:
                    analysis_df = latest_df[[target_col] + feature_cols].dropna()
                    if len(analysis_df) > 50:
                        st.success(f"現在選択されている項目がすべて揃っている銘柄: {len(analysis_df)}件")
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_corr = px.imshow(analysis_df.corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
                            st.plotly_chart(fig_corr, use_container_width=True)
                        with col2:
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(analysis_df[feature_cols], analysis_df[target_col])
                            importance_df = pd.DataFrame({'項目': feature_cols, '影響度': model.feature_importances_}).sort_values('影響度')
                            fig_imp = px.bar(importance_df, x='影響度', y='項目', orientation='h', color='影響度')
                            st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.warning(f"有効なデータが {len(analysis_df)} 件しかありません。空欄の多い項目を外してください。")
        else:
            st.info("Yahoo Financeから各銘柄の過去の株価を取得し、「直近〇ヶ月の株価上昇率」を算出して要因分析を行います。")
            col_period, col_features = st.columns([1, 2])
            with col_period:
                period_label = st.selectbox("上昇率の測定期間:", ["1ヶ月", "3ヶ月", "6ヶ月", "1年"])
                period_map = {"1ヶ月": "1mo", "3ヶ月": "3mo", "6ヶ月": "6mo", "1年": "1y"}
                period_val = period_map[period_label]
                target_col = f"株価上昇率 ({period_label})"
            with col_features:
                default_feats = [f for f in numeric_cols if "ROE" in f or "PER" in f or "自己資本比率" in f][:3]
                feature_cols = st.multiselect("説明変数を選択:", numeric_cols, default=default_feats)

            if st.button("🚀 株価データを取得して解析開始"):
                with st.spinner(f"全銘柄の過去{period_label}の株価を一括取得中..."):
                    tickers = [get_ticker_symbol(code) for code in latest_df['コード'].dropna().unique()]
                    try:
                        data = yf.download(tickers, period=period_val, group_by="ticker", threads=False)
                        returns = {}
                        for ticker in tickers:
                            try:
                                df_t = data if len(tickers) == 1 else data[ticker]
                                df_t = df_t.dropna(subset=['Close'])
                                if len(df_t) >= 2:
                                    start_price = df_t['Close'].iloc[0]
                                    end_price = df_t['Close'].iloc[-1]
                                    returns[ticker] = (end_price - start_price) / start_price * 100
                            except: pass
                        
                        analysis_target_df = latest_df.copy()
                        analysis_target_df[target_col] = analysis_target_df['コード'].apply(lambda x: returns.get(get_ticker_symbol(x)))
                        analysis_df = analysis_target_df[[target_col] + feature_cols].dropna()
                        
                        if len(analysis_df) > 50:
                            st.success(f"解析に使用した有効銘柄: {len(analysis_df)}件")
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_corr = px.imshow(analysis_df.corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
                                st.plotly_chart(fig_corr, use_container_width=True)
                            with col2:
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                                model.fit(analysis_df[feature_cols], analysis_df[target_col])
                                importance_df = pd.DataFrame({'項目': feature_cols, '影響度': model.feature_importances_}).sort_values('影響度')
                                fig_imp = px.bar(importance_df, x='影響度', y='項目', orientation='h', color='影響度')
                                st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.warning("有効データ不足です。空欄の多い項目を外してください。")
                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")

    # --- モード4: バックテスト ---
    elif mode == "⏱️ 過去データでの答え合わせ (バックテスト)":
        st.subheader("過去の指標と実際のリターンの答え合わせ")
        past_price_col = st.selectbox("「当時の株価」の列:", [c for c in numeric_cols if "値" in c or "株価" in c], index=0)
        available_features = [c for c in numeric_cols if c != past_price_col]
        feature_cols = st.multiselect("検証したい当時の指標:", available_features, default=[f for f in available_features if "ROE" in f][:1])

        if st.button("🚀 最新株価を取得して検証開始"):
            with st.spinner("最新株価を取得中..."):
                target_df = latest_df.copy()
                tickers = [get_ticker_symbol(code) for code in target_df['コード'].dropna().unique()]
                
                try:
                    data = yf.download(tickers, period="1d", group_by="ticker", threads=False)
                    current_prices = {}
                    for ticker in tickers:
                        try:
                            if len(tickers) == 1:
                                current_prices[ticker] = data['Close'].iloc[-1]
                            else:
                                current_prices[ticker] = data[ticker]['Close'].iloc[-1]
                        except: pass
                    
                    target_df['最新株価'] = target_df['コード'].apply(lambda x: current_prices.get(get_ticker_symbol(x)))
                    target_df['実際の上昇率(%)'] = (target_df['最新株価'] - target_df[past_price_col]) / target_df[past_price_col] * 100
                    analysis_df = target_df[['実際の上昇率(%)'] + feature_cols].dropna()
                    
                    if len(analysis_df) > 50:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.dataframe(target_df[['コード', '銘柄名', past_price_col, '最新株価', '実際の上昇率(%)']].dropna().sort_values('実際の上昇率(%)', ascending=False).head(10))
                        with col2:
                            model = RandomForestRegressor(random_state=42)
                            model.fit(analysis_df[feature_cols], analysis_df['実際の上昇率(%)'])
                            st.plotly_chart(px.bar(pd.DataFrame({'項目': feature_cols, '影響度': model.feature_importances_}).sort_values('影響度'), x='影響度', y='項目', orientation='h', color='影響度'), use_container_width=True)
                    else:
                        st.warning("有効データ不足です。空欄の多い項目を外してください。")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

else:
    st.error("⚠️ `master_database.csv` が見つかりません。アプリ (`分析ソフト.py`) と同じフォルダにファイルを配置してください。")
