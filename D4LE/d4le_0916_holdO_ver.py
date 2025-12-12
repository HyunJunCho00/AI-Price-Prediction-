import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate

# ====================================
# 1. 설정 (Hyperparameters)
# ====================================
MARKETS = ["KRW-ETH", "USDT-ETH", "BTC-ETH"]
# 논문에 명시된 최적의 하이퍼파라미터
LOOK_BACK = 7           # Window Size
EPOCHS = 50             # Epochs
LSTM_UNITS = 32         # Number of LSTM Neurons
BATCH_SIZE = 32
TRAIN_RATIO = 0.8

# 방향성 결정 임계값 (%)
LONG_THRESHOLD = 1.0
SHORT_THRESHOLD = -1.0

# ====================================
# 2. 데이터 로드 및 통합
# ====================================
def load_and_prepare_data(markets):
    """3개 마켓 OHLCV 데이터를 로드하고 통합합니다."""
    dfs = {}
    for market in markets:
        try:
            df = pd.read_csv(f"{market}_ohlcv.csv", index_col=0, parse_dates=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            prefix = f"{market.split('-')[0].lower()}_"
            df = df.add_prefix(prefix)
            dfs[market] = df
        except FileNotFoundError:
            print(f"오류: {market}_ohlcv.csv 파일을 찾을 수 없습니다.")
            return None

    main_df = dfs["USDT-ETH"]
    main_df = main_df.join(dfs["KRW-ETH"], how='inner')
    main_df = main_df.join(dfs["BTC-ETH"], how='inner')
    
    print(f"데이터 통합 완료. 총 {len(main_df.columns)}개의 특성(Feature) 생성.")
    return main_df

# ====================================
# 3. LSTM용 데이터셋 구축
# ====================================
def create_lstm_dataset(df):
    """LSTM 학습을 위한 시퀀스 데이터셋을 생성합니다."""
    df = df.copy().dropna()
    
    y = df['usdt_close'].shift(-1)
    X = df
    
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - LOOK_BACK):
        X_seq.append(X_scaled[i:i + LOOK_BACK])
        y_seq.append(y_scaled[i + LOOK_BACK])
        
    return np.array(X_seq), np.array(y_seq), scaler_X, scaler_y, X.index[LOOK_BACK:]

# ====================================
# 4. D4LE 모델 구축 (논문 기반)
# ====================================
def build_d4le_model(input_shape, lstm_units):
    """논문에 기술된 4개의 LSTM을 앙상블하는 D4LE 모델을 구축합니다."""
    input_layer = Input(shape=input_shape)

    lstm1 = LSTM(units=lstm_units, activation='tanh', return_sequences=False)(input_layer)
    lstm2 = LSTM(units=lstm_units, activation='tanh', return_sequences=False)(input_layer)
    lstm3 = LSTM(units=lstm_units, activation='tanh', return_sequences=False)(input_layer)
    lstm4 = LSTM(units=lstm_units, activation='tanh', return_sequences=False)(input_layer)

    merged = Concatenate()([lstm1, lstm2, lstm3, lstm4])
    dense_layer = Dense(64, activation='relu')(merged)
    output_layer = Dense(1)(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("D4LE 모델 구축 완료.")
    model.summary()
    return model

# ====================================
# 5. 모델 성능 평가 함수 (추가된 부분)
# ====================================
def evaluate_model_performance(y_true_price, y_pred_price, y_true_labels, y_pred_labels):
    """RMSE와 상세 분류 지표를 계산하고 출력합니다."""
    
    print("\n" + "="*50)
    print("모델 종합 성능 평가")
    print("="*50)

    # 1. RMSE 계산 (회귀 성능)
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    nrmse = rmse / np.mean(y_true_price) * 100
    print(f"가격 예측 성능 (RMSE): {rmse:,.2f} USDT")
    print(f"정규화된 RMSE: {nrmse:.2f}% (값이 낮을수록 좋음)")
    print("-" * 50)
    
    # 2. 분류 성능 평가
    print("방향성 예측 성능 (Classification Metrics)\n")
    
    report = classification_report(y_true_labels, y_pred_labels, 
                                   target_names=['SHORT', 'HOLD', 'LONG'], output_dict=True, zero_division=0)
    accuracy = report['accuracy']
    
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    results = {}
    for i, label in enumerate(['SHORT', 'HOLD', 'LONG']):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0 # Recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        results[label] = {
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision
        }
    
    print(f"{'Metric':<12} | {'SHORT':<10} | {'HOLD':<10} | {'LONG':<10} | {'Average':<10}")
    print("-" * 60)
    
    metrics_to_print = ['Sensitivity', 'Specificity', 'Precision']
    for metric in metrics_to_print:
        short_val = results['SHORT'][metric]
        hold_val = results['HOLD'][metric]
        long_val = results['LONG'][metric]
        avg_val = np.mean([short_val, hold_val, long_val])
        print(f"{metric:<12} | {short_val:<10.4f} | {hold_val:<10.4f} | {long_val:<10.4f} | {avg_val:<10.4f}")

    f1_avg = report['macro avg']['f1-score']
    print(f"{'F1 Score':<12} | {'-':<10} | {'-':<10} | {'-':<10} | {f1_avg:<10.4f}")
    print(f"{'Accuracy':<12} | {'-':<10} | {'-':<10} | {'-':<10} | {accuracy*100:<9.2f}%")
    print("="*50)
    
    return cm

# ====================================
# 6. 메인 실행 로직
# ====================================
unified_df = load_and_prepare_data(MARKETS)
if unified_df is not None:
    X_seq, y_seq, scaler_X, scaler_y, dates = create_lstm_dataset(unified_df)

    split_idx = int(len(X_seq) * TRAIN_RATIO)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    dates_test = dates[split_idx:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    d4le_model = build_d4le_model(input_shape, LSTM_UNITS)
    
    print("\n모델 훈련 시작...")
    history = d4le_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                             validation_split=0.1, verbose=1)
    print("모델 훈련 완료.")

    # 테스트 데이터 예측
    predicted_scaled = d4le_model.predict(X_test)
    predicted_prices = scaler_y.inverse_transform(predicted_scaled)
    actual_prices = scaler_y.inverse_transform(y_test)

    # --- 평가를 위한 방향성 라벨 생성 ---
    # 테스트 기간의 '어제' 종가 데이터 추출
    test_start_date_loc = unified_df.index.get_loc(dates_test[0])
    last_actual_prices_test = unified_df['usdt_close'].iloc[test_start_date_loc - 1 : test_start_date_loc - 1 + len(dates_test)].values
    
    predicted_returns = (predicted_prices.flatten() - last_actual_prices_test) / last_actual_prices_test * 100
    actual_returns = (actual_prices.flatten() - last_actual_prices_test) / last_actual_prices_test * 100

    def get_label(returns):
        labels = np.ones_like(returns, dtype=int) # HOLD
        labels[returns > LONG_THRESHOLD] = 2    # LONG
        labels[returns < SHORT_THRESHOLD] = 0   # SHORT
        return labels

    y_pred_labels = get_label(predicted_returns)
    y_true_labels = get_label(actual_returns)

    # --- 새로운 평가 함수 호출 ---
    cm = evaluate_model_performance(actual_prices, predicted_prices, y_true_labels, y_pred_labels)
    
    # --- 시각화 ---
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(15, 7))
    plt.plot(dates_test, actual_prices, color='cyan', label='Original Price')
    plt.plot(dates_test, predicted_prices, color='gray', label='Predicted Price')
    plt.title('D4LE Model: Actual vs. Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['SHORT', 'HOLD', 'LONG'], yticklabels=['SHORT', 'HOLD', 'LONG'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # --- 내일 방향성 예측 ---
    print("\n" + "="*50)
    print("내일 ETH 가격 방향성 예측")
    print("="*50)
    
    last_sequence_scaled = X_seq[-1].reshape(1, LOOK_BACK, X_seq.shape[2])
    tomorrow_pred_scaled = d4le_model.predict(last_sequence_scaled)
    tomorrow_pred_price = scaler_y.inverse_transform(tomorrow_pred_scaled)[0][0]
    
    last_actual_price = unified_df['usdt_close'].iloc[-1]
    predicted_return = (tomorrow_pred_price - last_actual_price) / last_actual_price * 100
    
    direction = "HOLD (유지)"
    if predicted_return > LONG_THRESHOLD:
        direction = "LONG (상승)"
    elif predicted_return < SHORT_THRESHOLD:
        direction = "SHORT (하락)"
        
    print(f"오늘 종가 (USDT): {last_actual_price:,.2f}")
    print(f"내일 예측 종가 (USDT): {tomorrow_pred_price:,.2f}")
    print(f"예상 수익률: {predicted_return:.2f}%")
    print("-" * 50)
    print(f"💡 최종 예측 방향: {direction}")
    print("="*50)
    print("\n*신뢰도는 모델의 예측 수익률 크기에 비례합니다. 절대적인 확률이 아닙니다.")
