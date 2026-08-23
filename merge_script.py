import pandas as pd
import glob
import os
from datetime import datetime

def accumulate_daily_data():
    # フォルダ内の各CSVファイルを自動検索
    base_file = glob.glob('*_全銘柄.csv')
    finance_file = glob.glob('*_財務関連.csv')
    consensus_file = glob.glob('*_コンセンサス情報.csv')

    if not (base_file and finance_file and consensus_file):
        print("エラー: 今日のCSVファイルがすべて揃っていません。")
        return

    # ファイル名から日付（先頭の数字）を抽出して「YYYY-MM-DD」形式に変換
    date_str = base_file[0].split('_')[0]
    dt = datetime.strptime(date_str, '%y%m%d')
    formatted_date = dt.strftime('%Y-%m-%d')
    print(f"{formatted_date} のデータを処理しています...")

    # CSVの読み込み（ハイフンを欠損値として処理）
    df_base = pd.read_csv(base_file[0], na_values=['-', '－'], encoding='utf-8-sig')
    df_finance = pd.read_csv(finance_file[0], na_values=['-', '－'], encoding='utf-8-sig')
    df_consensus = pd.read_csv(consensus_file[0], na_values=['-', '－'], encoding='utf-8-sig')

    # 【重要1】楽天証券のCSVは列名に余計なスペースが入ることがあるため、全データの列名の空白を取り除く
    for df in [df_base, df_finance, df_consensus]:
        df.columns = df.columns.str.strip()

    # --- 重複列を排除して結合するロジック ---
    df_merged = df_base.copy()

    # 【重要2】df_merged（ベース）に既に存在する列（銘柄名や市場など）を特定し、財務データ側から除外する（'コード'は結合キーなので残す）
    cols_to_use_finance = df_finance.columns.difference(df_merged.columns).tolist() + ['コード']
    df_merged = pd.merge(df_merged, df_finance[cols_to_use_finance], on='コード', how='left')

    # 【重要3】同様に、現在のdf_mergedに存在する列をコンセンサスデータ側から除外して結合
    cols_to_use_consensus = df_consensus.columns.difference(df_merged.columns).tolist() + ['コード']
    df_merged = pd.merge(df_merged, df_consensus[cols_to_use_consensus], on='コード', how='left')

    # 銘柄コードの重複行を削除（念のため）
    df_merged = df_merged.drop_duplicates(subset=['コード'])
    
    # 取得日を先頭列に追加
    df_merged.insert(0, '取得日', formatted_date)

    # --- マスターファイルへの追記保存 ---
    master_file = 'master_database.csv'
    
    if os.path.exists(master_file):
        df_merged.to_csv(master_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"既存の {master_file} にデータを追加しました！")
    else:
        df_merged.to_csv(master_file, index=False, encoding='utf-8-sig')
        print(f"新規に {master_file} を作成し、データを保存しました！")

if __name__ == "__main__":
    accumulate_daily_data()