#!/usr/bin/env python3
"""
社區篩檢個案返診動機預測模型
根據四個關鍵因素預測個案的返診意願（紅/黃/綠燈）

因素權重（依重要性排序）：
1. 身心障礙證明（有 → 高紅燈）
2. 年齡（年輕 → 高紅燈）
3. 交通工具（無 → 高紅燈）
4. 主要照顧者（女兒 → 高紅燈）

Author: 洪士涵
Date: 2026-03-14
"""

import pandas as pd
import numpy as np

# 權重配置（可調整）
WEIGHTS = {
    '身心障礙': {'有': 2.0, '無': 0},
    '年齡': {'75歲以下': 2.5, '75-84歲': 1.0, '85歲以上': 0},
    '交通工具': {'無': 2.0, '有': 0},
    '主要照顧者': {'女兒': 1.5, '同居者': 1.5, '案夫': 1.0, '案妻': 0.5, '案子': 0, '兒子': 0, '孩子': 0, '自己': 0},
    '疾病': {'未知': 2.0, '中風/CVA': 1.5, '心臟病': 1.0, '糖尿病': 0.5, '高血壓': 0},
    '性別': {'男': 0.5, '女': 0},
    '獨居': {'是': 0.5, '否': 0},
}

def calculate_risk_score(row):
    """計算風險分數（0-10）"""
    score = 0
    
    # 身心障礙
    if row.get('身心殘障證明') in WEIGHTS['身心障礙']:
        score += WEIGHTS['身心障礙'][row['身心殘障證明']]
    
    # 年齡
    if pd.notna(row.get('年齡')):
        age = row['年齡']
        if age < 75:
            score += WEIGHTS['年齡']['75歲以下']
        elif age < 85:
            score += WEIGHTS['年齡']['75-84歲']
        else:
            score += WEIGHTS['年齡']['85歲以上']
    
    # 交通工具
    if row.get('有無交通工具') in WEIGHTS['交通工具']:
        score += WEIGHTS['交通工具'][row['有無交通工具']]
    
    # 主要照顧者
    if row.get('主要照顧者') in WEIGHTS['主要照顧者']:
        score += WEIGHTS['主要照顧者'][row['主要照顧者']]
    
    # 疾病
    disease = str(row.get('疾病', '')).upper()
    if '未知' in disease or pd.isna(row.get('疾病')):
        score += WEIGHTS['疾病']['未知']
    elif 'CVA' in disease or '中風' in disease:
        score += WEIGHTS['疾病']['中風/CVA']
    elif '心臟' in disease or '心血管' in disease:
        score += WEIGHTS['疾病']['心臟病']
    elif 'DM' in disease or '糖尿病' in disease:
        score += WEIGHTS['疾病']['糖尿病']
    elif 'HTN' in disease or '高血壓' in disease:
        score += WEIGHTS['疾病']['高血壓']
    
    # 性別
    if row.get('性別') in WEIGHTS['性別']:
        score += WEIGHTS['性別'][row['性別']]
    
    # 獨居
    if row.get('是否為獨居') in WEIGHTS['獨居']:
        score += WEIGHTS['獨居'][row['是否為獨居']]
    
    return score

def predict_light(row):
    """根據風險分數預測燈號"""
    score = calculate_risk_score(row)
    
    if score >= 3.0:
        return '🔴 紅燈'
    elif score >= 1.5:
        return '🟡 黃燈'
    else:
        return '🟢 綠燈'

def get_priority_score(row):
    """取得優先追蹤分數（越高越需要關懷）"""
    return calculate_risk_score(row)

def predict_from_input():
    """互動式輸入預測"""
    print("\n=== 社區篩檢個案返診動機預測 ===")
    print("請輸入個案資料：\n")
    
    data = {}
    data['身心殘障證明'] = input("身心障礙證明（有/無）: ").strip()
    age = input("年齡: ").strip()
    data['年齡'] = float(age) if age else np.nan
    data['有無交通工具'] = input("交通工具（有/無）: ").strip()
    data['主要照顧者'] = input("主要照顧者: ").strip()
    data['疾病'] = input("疾病: ").strip()
    data['性別'] = input("性別（男/女）: ").strip()
    data['是否為獨居'] = input("是否獨居（是/否）: ").strip()
    
    score = calculate_risk_score(data)
    prediction = predict_light(data)
    
    print(f"\n=== 預測結果 ===")
    print(f"風險分數: {score:.1f} / 10")
    print(f"預測燈號: {prediction}")
    
    if score >= 3.0:
        print("\n⚠️ 建議：需要積極追蹤關懷")
    elif score >= 1.5:
        print("\n💡 建議：需要持續關懷")
    else:
        print("\n✅ 建議：定期關懷即可")

def analyze_excel(file_path):
    """分析 Excel 檔案"""
    df = pd.read_excel(file_path, sheet_name='rawdata')
    
    # 計算風險分數
    df['風險分數'] = df.apply(calculate_risk_score, axis=1)
    df['預測燈號'] = df.apply(predict_from_input, axis=1)
    
    # 排序（高風險在前）
    df_sorted = df.sort_values('風險分數', ascending=False)
    
    return df_sorted

def main():
    import sys
    
    if len(sys.argv) > 1:
        # 有參數，視為 Excel 檔案路徑
        file_path = sys.argv[1]
        print(f"分析檔案: {file_path}")
        result = analyze_excel(file_path)
        print("\n=== 分析結果 ===")
        print(result[['姓名', '風險分數', '預測燈號', '主要照顧者', '年齡']].head(10))
    else:
        # 無參數，進入互動模式
        predict_from_input()

if __name__ == "__main__":
    main()
