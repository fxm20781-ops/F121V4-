import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. 網頁初始化配置
st.set_page_config(page_title="F121 製程最佳化控制系統", layout="wide")
st.title("🏭 F121 天然氣最低消耗控制系統 (多目標黃金平衡版)")

# 2. 檔案上傳元件
uploaded_file = st.file_uploader("請上傳您的 F121 歷史數據 Excel 檔 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # 讀取 Excel 檔案，跳過第二行 Tag 行
        df = pd.read_excel(uploaded_file, skiprows=[1])
        df.columns = df.columns.str.strip()
        
        # 欄位定義：5 個自變數 (X)
        feature_fixed = ['DT operation', 'C141 operation', 'F121outlet temperature']
        feature_controllable = ['F121 CLO circulation flow', 'F121 Oxygen content %']
        all_features = feature_fixed + feature_controllable
        
        target_ng = 'F121 NG consumption'
        target_temp = 'C122 bottom temperature'
        
        required_cols = all_features + [target_ng, target_temp]
        
        # 清理並轉換數據
        df_clean = df[required_cols].dropna().apply(pd.to_numeric, errors='coerce').dropna()
        
        if df_clean.empty:
            st.error("❌ 經篩選後沒有有效的數據，請確認 Excel 內的數值是否正確。")
        else:
            X = df_clean[all_features]
            y_ng = df_clean[target_ng]
            y_temp = df_clean[target_temp]
            
            # 獲取各欄位歷史極值
            hist_temp_min = float(df_clean[target_temp].min())
            hist_temp_max = float(df_clean[target_temp].max())
            
            # 訓練二次多項式模型
            @st.cache_resource
            def train_and_get_coefs(_X, _y_ng, _y_temp):
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(_X)
                
                m_ng = LinearRegression().fit(X_poly, _y_ng)
                m_temp = LinearRegression().fit(X_poly, _y_temp)
                return poly, m_ng, m_temp
                
            poly_transformer, model_ng, model_temp = train_and_get_coefs(X, y_ng, y_temp)
            st.success("✅ Excel 數據載入成功，AI 控制核心已準備就緒！")
            
            # 3. 側邊欄：固定輸入項目與【全新優化】操作穩定度滑桿
            st.sidebar.header("📋 當前固定輸入/排程條件")
            input_dt = st.sidebar.number_input("DT operation 稼動率", value=0.80, step=0.01)
            input_c141 = st.sidebar.number_input("C141 operation 稼動率", value=1.20, step=0.01)
            input_out_temp = st.sidebar.number_input("F121 outlet temperature 出口溫度 (°C)", value=332.0, step=0.1)

            st.sidebar.markdown("---")
            st.sidebar.header("⚖️ 現場操作穩定權重 (核心調整)")
            st.sidebar.write("提高下方權重，AI 推薦值會更傾向於留在『歷史最穩定的中間區間』，不再死卡上下限：")
            
            stability_weight = st.sidebar.slider(
                "現場操作穩定度權重 (擺脫邊界)", 
                min_value=0.0, 
                max_value=100.0, 
                value=15.0,  # 預設給予 15.0 的柔性穩定引導
                step=0.5
            )

            # 4. 主畫面：控制限制範圍與下游客層溫度約束
            st.header("⚙️ 設定操作安全限制範圍與製程約束")
            
            st.markdown("### 🌡️ 下游 C122 塔底溫度安全約束範圍")
            st.write(f"歷史實測操作區間為：{hist_temp_min:.1f} °C ~ {hist_temp_max:.1f} °C")
            c_temp_min = st.number_input("強制約束 - 塔底溫度下限 (°C)", value=300.0, step=0.1)
            c_temp_max = st.number_input("強制約束 - 塔底溫度上限 (°C)", value=308.0, step=0.1)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### CLO Circulation Flow")
                clo_min = st.number_input("安全下限", value=50.0, step=0.1, key="clo_min")
                clo_max = st.number_input("安全上限", value=56.0, step=0.1, key="clo_max")
                # 計算流量中心點做為黃金基準
                clo_center = (clo_min + clo_max) / 2.0
            with col2:
                st.markdown("### Oxygen Content %")
                ox_min = st.number_input("安全下限 (%)", value=3.0, step=0.1, key="ox_min")
                ox_max = st.number_input("安全上限 (%)", value=7.0, step=0.1, key="ox_max")

            # 5. 黃金多目標動態權衡搜尋
            if st.button("🚀 開始計算最低天然氣消耗控制策略", type="primary"):
                grid_clo = np.linspace(clo_min, clo_max, 100) # 切細到 100 點更精準
                grid_ox = np.linspace(ox_min, ox_max, 100)
                
                c_mesh, o_mesh = np.meshgrid(grid_clo, grid_ox)
                flat_clo = c_mesh.ravel()
                flat_ox = o_mesh.ravel()
                
                flat_dt = np.full_like(flat_clo, input_dt)
                flat_c141 = np.full_like(flat_clo, input_c141)
                flat_out_temp = np.full_like(flat_clo, input_out_temp)
                
                raw_features = np.column_stack([flat_dt, flat_c141, flat_out_temp, flat_clo, flat_ox])
                test_features_poly = poly_transformer.transform(raw_features)
                
                # 預測天然氣與 C122 溫度
                pred_ng_all = model_ng.predict(test_features_poly)
                pred_temp_all = model_temp.predict(test_features_poly)
                
                # 綜合得分計算
                total_score = pred_ng_all.copy()
                for i in range(len(total_score)):
                    temp = pred_temp_all[i]
                    # 1. 硬性溫度超標處罰 (5000點)
                    if temp < c_temp_min:
                        total_score[i] += (c_temp_min - temp) * 5000
                    elif temp > c_temp_max:
                        total_score[i] += (temp - c_temp_max) * 5000
                    
                    # 2. 【核心新增】偏離現場操作中心點 (53.0) 的柔性處罰
                    # 當流量偏離中心越遠，處罰分數越高，強迫 AI 尋找中間的操作甜蜜點
                    deviation = (flat_clo[i] - clo_center) ** 2
                    total_score[i] += deviation * stability_weight
                
                best_idx = np.argmin(total_score)
                
                opt_clo = flat_clo[best_idx]
                opt_ox = flat_ox[best_idx]
                min_ng_consumption = pred_ng_all[best_idx]
                predicted_c122_temp = pred_temp_all[best_idx]
                
                # 6. 結果呈現
                st.markdown("---")
                st.subheader("🎯 最佳化控制推薦結果")
                
                m1, m2 = st.columns(2)
                m1.metric(label="📉 預期最低 F121 NG Consumption (Y)", value=f"{min_ng_consumption:.2f}")
                m2.metric(label="🌡️ 同時預測 C122 Bottom Temperature", value=f"{predicted_c122_temp:.2f} °C")
                
                recommend_df = pd.DataFrame({
                    "製程控制項目": ["F121 CLO circulation flow", "F121 Oxygen content %"],
                    "💡 最佳推薦控制值": [f"{opt_clo:.2f}", f"{opt_ox:.2f} %"],
                    "當前設定安全操作範圍": [f"{clo_min} ~ {clo_max}", f"{ox_min} ~ {ox_max}"]
                })
                st.table(recommend_df)
                
                # 7. 自動導出預測公式
                st.markdown("---")
                st.subheader("📝 F121 天然氣消耗量 (NG) 的二次迴歸預測公式")
                
                intercept = model_ng.intercept_
                coefs = model_ng.coef_
                feature_names = poly_transformer.get_feature_names_out(all_features)
                
                rename_dict = {
                    'DT operation': 'DT', 'C141 operation': 'C141',
                    'F121outlet temperature': 'Temp_out', 'F121 CLO circulation flow': 'CLO_flow',
                    'F121 Oxygen content %': 'Oxygen'
                }
                
                formula_text = f"**NG 消耗量** = {intercept:.4f}\n"
                for coef, name in zip(coefs, feature_names):
                    display_name = name
                    for orig, short in rename_dict.items():
                        display_name = display_name.replace(orig, short)
                    display_name = display_name.replace(" ", " × ")
                    if coef >= 0:
                        formula_text += f" + ({coef:.6f} × {display_name})\n"
                    else:
                        formula_text += f" - ({abs(coef):.6f} × {display_name})\n"
                st.code(formula_text, language="text")
                
                # 8. 自動原因診斷分析
                st.markdown("---")
                st.subheader("🔍 控制尋優卡點診斷分析")
                st.info(f"💡 **成功解鎖操作點！** 當前設定的『現場操作穩定度權重』為 `{stability_weight}`，目前的推薦值 `{opt_clo:.2f}` 已成功結合能耗效益與現場穩定性。如果您希望推薦值更靠近中心點 (53.0)，可以試著將側邊欄滑桿**往右調大**；若希望更追求極致省能而不在乎卡邊界，則可**往左調小**。")
                        
    except Exception as e:
        st.error(f"❌ 計算時發生錯誤: {e}")
else:
    st.info("💡 請在上方直接上傳您的原始 F121 數據 Excel (.xlsx) 檔案以啟動系統。")
