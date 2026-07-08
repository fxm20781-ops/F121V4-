import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. 網頁初始化配置
st.set_page_config(page_title="F121 製程最佳化控制系統", layout="wide")
st.title("🏭 F121 天然氣最低消耗控制系統 (完全自訂初始與邊界版)")

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
            
            # 訓練二次多項式模型
            @st.cache_resource
            def train_and_get_coefs(_X, _y_ng, _y_temp):
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(_X)
                
                m_ng = LinearRegression().fit(X_poly, _y_ng)
                m_temp = LinearRegression().fit(X_poly, _y_temp)
                return poly, m_ng, m_temp
                
            poly_transformer, model_ng, model_temp = train_and_get_coefs(X, y_ng, y_temp)
            st.success("✅ Excel 數據載入成功，AI 二次公式已擬合完成！")
            
            # 3. 側邊欄：固定輸入項目，全部帶入您指定的初始預設值
            st.sidebar.header("📋 當前固定輸入/排程條件")
            
            input_dt = st.sidebar.number_input(
                "DT operation 稼動率", 
                value=0.80,  # 初始預設值固定為 0.8
                step=0.01
            )
            input_c141 = st.sidebar.number_input(
                "C141 operation 稼動率", 
                value=1.20,  # 初始預設值固定為 1.2
                step=0.01
            )
            input_out_temp = st.sidebar.number_input(
                "F121 outlet temperature 出口溫度 (°C)", 
                value=332.0,  # 初始預設值固定為 332
                step=0.1
            )

            # 4. 【關鍵修改】主畫面：設定可控參數操作限制，帶入您指定的 50~56 與 3~7 預設值
            st.header("⚙️ 設定可控參數的操作安全限制範圍 (Safety Bounds)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### CLO Circulation Flow")
                clo_min = st.number_input("安全下限", value=50.0, step=0.1, key="clo_min")
                clo_max = st.number_input("安全上限", value=56.0, step=0.1, key="clo_max")
            with col2:
                st.markdown("### Oxygen Content %")
                ox_min = st.number_input("安全下限 (%)", value=3.0, step=0.1, key="ox_min")
                ox_max = st.number_input("安全上限 (%)", value=7.0, step=0.1, key="ox_max")

            # 5. 極速 2D 網格最佳化搜尋
            if st.button("🚀 開始計算最低天然氣消耗控制策略", type="primary"):
                grid_clo = np.linspace(clo_min, clo_max, 30)
                grid_ox = np.linspace(ox_min, ox_max, 30)
                
                c_mesh, o_mesh = np.meshgrid(grid_clo, grid_ox)
                flat_clo = c_mesh.ravel()
                flat_ox = o_mesh.ravel()
                
                flat_dt = np.full_like(flat_clo, input_dt)
                flat_c141 = np.full_like(flat_clo, input_c141)
                flat_out_temp = np.full_like(flat_clo, input_out_temp)
                
                raw_features = np.column_stack([flat_dt, flat_c141, flat_out_temp, flat_clo, flat_ox])
                test_features_poly = poly_transformer.transform(raw_features)
                
                pred_ng_all = model_ng.predict(test_features_poly)
                best_idx = np.argmin(pred_ng_all)
                
                opt_clo = flat_clo[best_idx]
                opt_ox = flat_ox[best_idx]
                min_ng_consumption = pred_ng_all[best_idx]
                
                best_raw_row = np.array([[input_dt, input_c141, input_out_temp, opt_clo, opt_ox]])
                best_poly_row = poly_transformer.transform(best_raw_row)
                predicted_c122_temp = model_temp.predict(best_poly_row)[0]
                
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
                st.caption("💡 變數名稱：DT=DT稼動率, C141=C141稼動率, Temp_out=出口溫度, CLO_flow=CLO流量, Oxygen=氧氣含量。")
                
                # 8. 自動分析為什麼 F121 CLO 流量會卡在上限
                st.markdown("---")
                st.subheader("🔍 CLO 流量卡上限原因診斷分析")
                
                idx_clo_1d = list(feature_names).index('F121 CLO circulation flow')
                idx_clo_2d = list(feature_names).index('F121 CLO circulation flow^2')
                
                coef_1d = coefs[idx_clo_1d]
                coef_2d = coefs[idx_clo_2d]
                
                st.write(f"在當前擬合的公式中：")
                st.write(f"* `CLO_flow` 的一次項係數為：`{coef_1d:.4f}`")
                st.write(f"* `CLO_flow^2` 的二次項係數為：`{coef_2d:.4f}`")
                
                if coef_2d < 0:
                    st.warning("⚠️ **診斷結果：曲線呈現倒烏龜殼（開口向下拋物線）**")
                    st.write(f"因為二次方項係數為負數，模型認為在整個範圍內流量開大能持續顯著節能，所以尋
