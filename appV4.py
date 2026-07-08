# 5. 多目標加權綜合評分搜尋 (防止死卡上下限)
            if st.button("🚀 開始計算最低天然氣消耗控制策略", type="primary"):
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
                
                # 預測天然氣消耗與 C122 溫度
                pred_ng_all = model_ng.predict(test_features_poly)
                pred_temp_all = model_temp.predict(test_features_poly)
                
                # 【柔性權衡邏輯】計算綜合處罰分數
                # 1. 基礎分數是天然氣消耗 (越低越好)
                total_score = pred_ng_all.copy()
                
                # 2. 如果預測溫度超出設定範圍，給予極大的數值懲罰 (強迫避開極端超溫/低溫)
                for i in range(len(total_score)):
                    temp = pred_temp_all[i]
                    if temp < c_temp_min:
                        # 溫度每低於下限 1 度，能耗懲罰增加 5000 點
                        total_score[i] += (c_temp_min - temp) * 5000
                    elif temp > c_temp_max:
                        # 溫度每高於上限 1 度，能耗懲罰增加 5000 點
                        total_score[i] += (temp - c_temp_max) * 5000
                
                # 找出綜合得分最低（最省能且最安全）的黃金操作點
                best_idx = np.argmin(total_score)
                
                opt_clo = flat_clo[best_idx]
                opt_ox = flat_ox[best_idx]
                min_ng_consumption = pred_ng_all[best_idx]
                predicted_c122_temp = pred_temp_all[best_idx]
