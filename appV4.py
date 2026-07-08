import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. 網頁初始化配置
st.set_page_config(page_title="F121 製程最佳化控制系統", layout="wide")
st.title("🏭 F121 天然氣最低消耗控制系統 (下游客層溫度約束尋優版)")

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
            
            # 獲取各欄位歷史極值與溫度的預設限制
            hist_temp_min = float(df_clean[target_temp].min())
            hist_temp_max = float(df_clean[target_temp].max())
            hist_temp_avg = float(df_clean[target_temp].mean())
            
            # 訓練二次多項式模型
            @st.cache_resource
            def train_and_get_coefs(_X, _y_ng, _y_temp):
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(_X)
                
                m_ng = LinearRegression().fit(X_poly, _y_ng)
                m_temp = LinearRegression().fit(X_poly, _y_temp)
                return poly, m_ng, m_temp
                
            poly_transformer, model_ng, model_temp = train_and_get_coefs(X, y_ng, y_temp)
            st.success("✅ Excel 數據載入成功，雙目標控制公式已擬合完成！")
            
            # 3. 側邊欄：固定輸入項目
            st.sidebar.header("📋 當前固定輸入/排程條件")
            input_dt = st.sidebar.number_input("DT operation 稼動率", value=0.80, step=0.01)
            input_c141 = st.sidebar.number_input("C141 operation 稼動率", value=1.20, step=0.01)
            input_out_temp = st.sidebar.number_input("F121 outlet temperature 出口溫度 (°C)", value=332.0, step=0.1)

            # 4. 主畫面：控制限制範圍與【新增】下游客層溫度約束
            st.header("⚙️ 設定操作安全限制範圍與製程約束")
            
            # 使用區塊包裝溫度約束
            st.markdown("### 🌡️ 下游 C122 塔底溫度安全約束 (Constraints)")
            st.write(f"歷史實測操作區間為：{hist_temp_min:.1f} °C ~ {hist_temp_max:.1f} °C")
            c_temp_min = st.number_input("強制約束 - 塔底溫度下限 (°C)", value=round(hist_temp_avg - 2, 1), step=0.1)
            c_temp_max = st.number_input("強制約束 - 塔底溫度上限 (°C)", value=round(hist_temp_avg + 2, 1), step=0.1)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### CLO Circulation Flow")
                clo_min = st.number_input("安全下限", value=50.0, step=0.1, key="clo_min")
                clo_max = st.number_input("安全上限", value=56.0, step=0.1, key="clo_max")
            with col2:
                st.markdown("### Oxygen Content %")
                ox_min = st.number_input("安全下限 (%)", value=3.0, step=0.1, key="ox_min")
                ox_max = st.number_input("安全上限 (%)", value=7.0, step=0.1, key="ox_max")

            # 5. 帶有溫度約束的 2D 網格最佳化搜尋
            if st.button("🚀 開始計算最低天然氣消耗控制策略", type="primary"):
                # 切細網格至 50x50 確保在邊界內找得更準
                grid_clo = np.linspace(clo_min, clo_max, 50)
                grid_ox = np.linspace(ox_min, ox_max, 50)
                
                c_mesh, o_mesh = np.meshgrid(grid_clo, grid_ox)
                flat_clo = c_mesh.ravel()
                flat_ox = o_mesh.ravel()
                
                flat_dt = np.full_like(flat_clo, input_dt)
                flat_c141 = np.full_like(flat_clo, input_c141)
                flat_out_temp = np.full_like(flat_clo, input_out_temp)
                
                raw_features = np.column_stack([flat_dt, flat_c141, flat_out_temp, flat_clo, flat_ox])
                test_features_poly = poly_transformer.transform(raw_features)
                
                # 1. 預測天然氣
                pred_ng_all = model_ng.predict(test_features_poly)
                # 2. 預測 C122 溫度
                pred_temp_all = model_temp.predict(test_features_poly)
                
                # 【關鍵邏輯】篩選出同時符合 C122 溫度限制的操作組合
                valid_mask = (pred_temp_all >= c_temp_min) & (pred_temp_all <= c_temp_max)
                
                if not np.any(valid_mask):
                    st.error(f"❌ 在目前的流量({clo_min}~{clo_max})範圍內，AI 無法找到能讓 C122 穩定維持在 {c_temp_min} ~ {c_temp_max} °C 的操作組合。請適度放寬溫度約束或操作上限。")
                else:
                    # 在符合安全溫度的組合中，尋找天然氣最低的索引
                    valid_indices = np.where(valid_mask)[0]
                    best_valid_sub_idx = np.argmin(pred_ng_all[valid_mask])
                    best_idx = valid_indices[best_valid_sub_idx]
                    
                    opt_clo = flat_clo[best_idx]
                    opt_ox = flat_ox[best_idx]
                    min_ng_consumption = pred_ng_all[best_idx]
                    predicted_c122_temp = pred_temp_all[best_idx]
                    
                    # 6. 結果呈現
                    st.markdown("---")
                    st.subheader("🎯 最佳化控制推薦結果 (已通過溫度安全約束)")
                    
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
                        
    except Exception as e:
        st.error(f"❌ 計算時發生錯誤: {e}")
else:
    st.info("💡 請在上方直接上傳您的原始 F121 數據 Excel (.xlsx) 檔案以啟動系統。")
