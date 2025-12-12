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
LOOK_BACK = 7
EPOCHS = 50
LSTM_UNITS = 32
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
LONG_THRESHOLD = 1.0
SHORT_THRESHOLD = -1.0

# ====================================
# 2. 데이터 로드 및 통합
# ====================================
def load_and_prepare_data(markets):
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
# 3. LSTM용 데이터셋 구축 - [수정] HOLD 데이터 제거
# ====================================
def create_binary_lstm_dataset(df):
    """상승/하락 데이터만 필터링하여 LSTM 데이터셋을 생성합니다."""
    df = df.copy().dropna()
    
    df['next_day_return'] = df['usdt_close'].pct_change().shift(-1) * 100
    
    # [수정] 라벨을 LONG(1)과 SHORT(0)으로만 지정
    df['direction_label'] = np.nan
    df.loc[df['next_day_return'] > LONG_THRESHOLD, 'direction_label'] = 1  # LONG
    df.loc[df['next_day_return'] < SHORT_THRESHOLD, 'direction_label'] = 0 # SHORT
    
    # [수정] HOLD에 해당하는 데이터(라벨이 없는 행)를 완전히 제거
    df_filtered = df.dropna(subset=['direction_label'])
    df_filtered['direction_label'] = df_filtered['direction_label'].astype(int)
    print(f"HOLD 데이터 제거 후 남은 데이터 수: {len(df_filtered)} (상승/하락이 명확한 날짜만 학습)")
    
    y = df_filtered['usdt_close'].shift(-1)
    X = df_filtered.drop(columns=['next_day_return', 'direction_label'])
    
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_seq, y_seq, original_indices = [], [], []
    for i in range(len(X_scaled) - LOOK_BACK):
        X_seq.append(X_scaled[i:i + LOOK_BACK])
        y_seq.append(y_scaled[i + LOOK_BACK])
        original_indices.append(y.index[i + LOOK_BACK])
        
    return np.array(X_seq), np.array(y_seq), scaler_X, scaler_y, pd.to_datetime(original_indices)


# ====================================
# 4. D4LE 모델 구축 (이전과 동일)
# ====================================
def build_d4le_model(input_shape, lstm_units):
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
# 5. 모델 성능 평가 함수 - [수정] 이진 분류용으로 변경
# ====================================
def evaluate_binary_model_performance(y_true_price, y_pred_price, y_true_labels, y_pred_labels):
    """RMSE와 이진 분류 지표를 계산하고 출력합니다."""
    print("\n" + "="*50)
    print("모델 종합 성능 평가")
    print("="*50)

    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    nrmse = rmse / np.mean(y_true_price) * 100
    print(f"가격 예측 성능 (RMSE): {rmse:,.2f} USDT")
    print(f"정규화된 RMSE: {nrmse:.2f}%")
    print("-" * 50)
    
    print("방향성 예측 성능 (Binary Classification Metrics)\n")
    print(classification_report(y_true_labels, y_pred_labels, target_names=['SHORT', 'LONG']))
    
    return confusion_matrix(y_true_labels, y_pred_labels)

# ====================================
# 6. 메인 실행 로직 - [수정] 이진 분류에 맞게 수정
# ====================================
unified_df = load_and_prepare_data(MARKETS)
if unified_df is not None:
    X_seq, y_seq, scaler_X, scaler_y, dates = create_binary_lstm_dataset(unified_df)

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

    predicted_scaled = d4le_model.predict(X_test)
    predicted_prices = scaler_y.inverse_transform(predicted_scaled)
    actual_prices = scaler_y.inverse_transform(y_test)

    # [수정] 이진 분류 라벨 생성
    test_start_date_loc = unified_df.index.get_loc(dates_test[0])
    last_actual_prices_test = unified_df['usdt_close'].loc[dates_test - pd.Timedelta(days=1)].values
    
    predicted_returns = (predicted_prices.flatten() - last_actual_prices_test) / last_actual_prices_test * 100
    actual_returns = (actual_prices.flatten() - last_actual_prices_test) / last_actual_prices_test * 100

    def get_binary_label(returns):
        labels = np.zeros_like(returns, dtype=int)
        labels[returns > 0] = 1 # LONG
        return labels

    y_pred_labels = get_binary_label(predicted_returns)
    y_true_labels = get_binary_label(actual_returns)

    # [수정] 이진 분류 평가 함수 호출
    cm = evaluate_binary_model_performance(actual_prices, predicted_prices, y_true_labels, y_pred_labels)
    
    # --- 시각화 ---
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(15, 7))
    plt.plot(dates_test, actual_prices, color='cyan', label='Original Price')
    plt.plot(dates_test, predicted_prices, color='gray', label='Predicted Price')
    plt.title('D4LE Model: Actual vs. Predicted Price (LONG/SHORT only)')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['SHORT', 'LONG'], yticklabels=['SHORT', 'LONG'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # --- 내일 방향성 예측 ---
    print("\n" + "="*50)
    print("내일 ETH 가격 방향성 예측")
    print("="*50)
    
    # [수정] 마지막 시퀀스 데이터 준비 시, 필터링되지 않은 원본 데이터프레임에서 가져와야 함
    last_sequence_data = unified_df.iloc[-LOOK_BACK:][[col for col in unified_df.columns if col not in ['next_day_return', 'direction_label']]]
    last_sequence_scaled = scaler_X.transform(last_sequence_data).reshape(1, LOOK_BACK, last_sequence_data.shape[1])
    
    tomorrow_pred_scaled = d4le_model.predict(last_sequence_scaled)
    tomorrow_pred_price = scaler_y.inverse_transform(tomorrow_pred_scaled)[0][0]
    
    last_actual_price = unified_df['usdt_close'].iloc[-1]
    predicted_return = (tomorrow_pred_price - last_actual_price) / last_actual_price * 100
    
    # [수정] HOLD 없이 LONG 또는 SHORT만 예측
    direction = "SHORT (하락)" if predicted_return < 0 else "LONG (상승)"
    confidence_proxy = abs(predicted_return)

    print(f"오늘 종가 (USDT): {last_actual_price:,.2f}")
    print(f"내일 예측 종가 (USDT): {tomorrow_pred_price:,.2f}")
    print(f"예상 수익률: {predicted_return:.2f}%")
    print("-" * 50)
    print(f"💡 최종 예측 방향: {direction}")
    print(f"(예상 변동폭 기반 신뢰도: {confidence_proxy:.2f}%)")
    print("=" * 50)
