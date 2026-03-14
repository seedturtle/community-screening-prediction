#!/usr/bin/env python3
"""
社區篩檢個案返診動機預測模型 - Flask Web API
根據四個關鍵因素預測個案的返診意願（紅/黃/綠燈）

部署方式：
1. pip install flask
2. python app.py
3. 部署到 Zeabur

Author: 洪士涵
Date: 2026-03-14
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np

app = Flask(__name__)

# 權重配置（可調整）
WEIGHTS = {
    '身心障礙': {'有': 2.0, '無': 0},
    '年齡': {'75歲以下': 2.5, '75-84歲': 1.0, '85歲以上': 0},
    '交通工具': {'無': 2.0, '有': 0},
    '主要照顧者': {'女兒': 1.5, '同居者': 1.5, '案夫': 1.0, '案妻': 0.5, '案子': 0, '兒子': 0, '孩子': 0, '自己': 0, '外籍看護': 0.5},
    '疾病': {'未知': 2.0, '中風/CVA': 1.5, '心臟病': 1.0, '糖尿病': 0.5, '高血壓': 0},
    '性別': {'男': 0.5, '女': 0},
    '獨居': {'是': 0.5, '否': 0},
}

def calculate_risk_score(data):
    """計算風險分數（0-10）"""
    score = 0
    
    # 身心障礙
    disability = data.get('身心障礙', '')
    if disability in WEIGHTS['身心障礙']:
        score += WEIGHTS['身心障礙'][disability]
    
    # 年齡
    age = data.get('年齡')
    if age is not None:
        try:
            age = float(age)
            if age < 75:
                score += WEIGHTS['年齡']['75歲以下']
            elif age < 85:
                score += WEIGHTS['年齡']['75-84歲']
            else:
                score += WEIGHTS['年齡']['85歲以上']
        except:
            pass
    
    # 交通工具
    transport = data.get('交通工具', '')
    if transport in WEIGHTS['交通工具']:
        score += WEIGHTS['交通工具'][transport]
    
    # 主要照顧者
    caregiver = data.get('主要照顧者', '')
    if caregiver in WEIGHTS['主要照顧者']:
        score += WEIGHTS['主要照顧者'][caregiver]
    
    # 疾病
    disease = str(data.get('疾病', '')).upper()
    if '未知' in disease or disease == '' or 'NAN' in disease:
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
    gender = data.get('性別', '')
    if gender in WEIGHTS['性別']:
        score += WEIGHTS['性別'][gender]
    
    # 獨居
    alone = data.get('是否獨居', '')
    if alone in WEIGHTS['獨居']:
        score += WEIGHTS['獨居'][alone]
    
    return round(score, 1)

def predict_light(score):
    """根據風險分數預測燈號"""
    if score >= 3.0:
        return {
            '燈號': '🔴 紅燈',
            '等級': '高風險',
            '建議': '需要積極追蹤關懷，建議儘速安排就醫'
        }
    elif score >= 1.5:
        return {
            '燈號': '🟡 黃燈',
            '等級': '中風險',
            '建議': '需要持續關懷，定期追蹤聯繫'
        }
    else:
        return {
            '燈號': '🟢 綠燈',
            '等級': '低風險',
            '建議': '回診意願高，定期關懷即可'
        }

# HTML 模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>社區篩檢個案返診動機預測</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'PingFang TC', 'Microsoft JhengHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 1.8em;
            margin-bottom: 10px;
        }
        .header p {
            opacity: 0.9;
            font-size: 0.9em;
        }
        .form {
            padding: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        select, input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            transition: all 0.3s;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102,126,234,0.3);
        }
        .btn:active {
            transform: translateY(0);
        }
        .result {
            display: none;
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
        }
        .result.red {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
            color: white;
        }
        .result.yellow {
            background: linear-gradient(135deg, #ffd93d 0%, #f5c842 100%);
            color: #333;
        }
        .result.green {
            background: linear-gradient(135deg, #6bcb77 0%, #4ecc6f 100%);
            color: white;
        }
        .result h2 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .result .score {
            font-size: 1.2em;
            margin-bottom: 15px;
            opacity: 0.9;
        }
        .result .suggestion {
            font-size: 1em;
            padding: 15px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
        }
        .info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .info h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .info ul {
            margin-left: 20px;
            color: #666;
        }
        .info li {
            margin-bottom: 5px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 社區篩檢個案返診動機預測</h1>
            <p>根據關鍵因素預測個案回診意願</p>
        </div>
        <div class="form">
            <div class="info">
                <h3>📋 燈號說明</h3>
                <ul>
                    <li><strong>🔴 紅燈</strong>：需要積極追蹤回診</li>
                    <li><strong>🟡 黃燈</strong>：需要持續關懷</li>
                    <li><strong>🟢 綠燈</strong>：回診意願高</li>
                </ul>
            </div>
            
            <div class="form-group">
                <label>年齡</label>
                <input type="number" id="age" placeholder="請輸入年齡" min="0" max="120">
            </div>
            
            <div class="form-group">
                <label>性別</label>
                <select id="gender">
                    <option value="">請選擇</option>
                    <option value="男">男</option>
                    <option value="女">女</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>身心障礙證明</label>
                <select id="disability">
                    <option value="">請選擇</option>
                    <option value="無">無</option>
                    <option value="有">有</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>交通工具</label>
                <select id="transport">
                    <option value="">請選擇</option>
                    <option value="有">有</option>
                    <option value="無">無</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>主要照顧者</label>
                <select id="caregiver">
                    <option value="">請選擇</option>
                    <option value="自己">自己</option>
                    <option value="案妻">案妻</option>
                    <option value="案夫">案夫</option>
                    <option value="案子">案子</option>
                    <option value="案女">案女</option>
                    <option value="女兒">女兒</option>
                    <option value="兒子">兒子</option>
                    <option value="孩子">孩子</option>
                    <option value="同居人">同居人</option>
                    <option value="外籍看護">外籍看護</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>疾病</label>
                <select id="disease">
                    <option value="">請選擇</option>
                    <option value="高血壓">高血壓</option>
                    <option value="糖尿病">糖尿病</option>
                    <option value="心臟病">心臟病</option>
                    <option value="中風/CVA">中風/CVA</option>
                    <option value="未知">未知</option>
                    <option value="其他">其他</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>是否獨居</label>
                <select id="alone">
                    <option value="">請選擇</option>
                    <option value="否">否</option>
                    <option value="是">是</option>
                </select>
            </div>
            
            <button class="btn" onclick="predict()">🔮 開始預測</button>
            
            <div id="result" class="result"></div>
        </div>
        <div class="footer">
            <p>資料來源：社區篩檢追蹤20260125範例</p>
        </div>
    </div>
    
    <script>
        async function predict() {
            const data = {
                '年齡': document.getElementById('age').value,
                '性別': document.getElementById('gender').value,
                '身心障礙': document.getElementById('disability').value,
                '交通工具': document.getElementById('transport').value,
                '主要照顧者': document.getElementById('caregiver').value,
                '疾病': document.getElementById('disease').value,
                '是否獨居': document.getElementById('alone').value
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = 'result ' + result.level;
                
                resultDiv.innerHTML = `
                    <h2>${result.燈號}</h2>
                    <div class="score">風險分數：${result.分數} / 10</div>
                    <div class="suggestion">${result.建議}</div>
                `;
            } catch (error) {
                alert('預測失敗，請稍後再試');
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    score = calculate_risk_score(data)
    prediction = predict_light(score)
    
    return jsonify({
        '分數': score,
        '燈號': prediction['燈號'],
        '等級': prediction['等級'],
        '建議': prediction['建議'],
        'level': 'red' if score >= 3.0 else ('yellow' if score >= 1.5 else 'green')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
