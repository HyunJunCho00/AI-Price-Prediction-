import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

# ====================================
# 1. 설정 (Hyperparameters)
# ====================================
MARKETS = ["KRW-ETH", "USDT-ETH", "BTC-ETH"]
LOOK_BACK = 30
EPOCHS = 100
LSTM_UNITS = 32
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
LONG_THRESHOLD = 1.0
SHORT_THRESHOLD = -1.0

# ====================================
# 2. 데이터 로드 및 통합 (기술적 지표 추가)
# ====================================
def load_and_prepare_data(markets):
    dfs = {}
    for market in markets:
        try:
            df = pd.read_csv(f"{market}_ohlcv.csv", index_col=0, parse_dates=True)
            prefix = f"{market.split('-')[0].lower()}_"
            df = df.add_prefix(prefix)
            dfs[market] = df
        except FileNotFoundError:
            print(f"오류: {market}_ohlcv.csv 파일을 찾을 수 없습니다.")
            return None

    main_df = dfs["USDT-ETH"].copy()
    main_df = main_df.join(dfs["KRW-ETH"], how='inner')
    main_df = main_df.join(dfs["BTC-ETH"], how='inner')
    
    print("기술적 지표 계산 중...")
    main_df.ta.sma(close=main_df['usdt_close'], length=10, append=True)
    main_df.ta.sma(close=main_df['usdt_close'], length=30, append=True)
    main_df.ta.rsi(close=main_df['usdt_close'], length=14, append=True)
    main_df.ta.macd(close=main_df['usdt_close'], fast=12, slow=26, signal=9, append=True)
    bbands = main_df.ta.bbands(close=main_df['usdt_close'], length=20, std=2)
    main_df = main_df.join(bbands)
    
    print(f"데이터 통합 및 특성 생성 완료. 총 {len(main_df.columns)}개의 특성(Feature) 생성.")
    return main_df

# ====================================
# 3. LSTM용 데이터셋 구축 (HOLD 데이터 제거)
# ====================================
def create_binary_lstm_dataset(df):
    df = df.copy().dropna()
    df['next_day_return'] = df['usdt_close'].pct_change().shift(-1) * 100
    
    df['direction_label'] = np.nan
    df.loc[df['next_day_return'] > LONG_THRESHOLD, 'direction_label'] = 1
    df.loc[df['next_day_return'] < SHORT_THRESHOLD, 'direction_label'] = 0
    
    df_filtered = df.dropna(subset=['direction_label'])
    df_filtered['direction_label'] = df_filtered['direction_label'].astype(int)
    print(f"HOLD 데이터 제거 후 남은 데이터 수: {len(df_filtered)} (상승/하락이 명확한 날짜만 학습)")
    
    y = df_filtered['usdt_close'].shift(-1)
    y_label = df_filtered['direction_label']
    # [수정] X를 정의할 때 타겟 관련 컬럼들을 모두 제거
    X = df_filtered.drop(columns=['next_day_return', 'direction_label'])
    
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    y_label = y_label.iloc[:-1]
    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_seq, y_seq, y_label_seq, original_indices = [], [], [], []
    for i in range(len(X_scaled) - LOOK_BACK):
        X_seq.append(X_scaled[i:i + LOOK_BACK])
        y_seq.append(y_scaled[i + LOOK_BACK])
        y_label_seq.append(y_label.iloc[i + LOOK_BACK])
        original_indices.append(y.index[i + LOOK_BACK])
        
    # [수정] 훈련에 사용된 특성 이름(X.columns)을 함께 반환
    return np.array(X_seq), np.array(y_seq), np.array(y_label_seq), scaler_X, scaler_y, X.columns, pd.to_datetime(original_indices)

# ====================================
# 4. D4LE 모델 구축
# ====================================
def build_d4le_model(input_shape, lstm_units):
    input_layer = Input(shape=input_shape)
    lstm1 = LSTM(units=lstm_units, activation='tanh')(input_layer)
    lstm2 = LSTM(units=lstm_units, activation='tanh')(input_layer)
    lstm3 = LSTM(units=lstm_units, activation='tanh')(input_layer)
    lstm4 = LSTM(units=lstm_units, activation='tanh')(input_layer)
    merged = Concatenate()([lstm1, lstm2, lstm3, lstm4])
    dense_layer = Dense(64, activation='relu')(merged)
    output_layer = Dense(1)(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("D4LE 모델 구축 완료.")
    return model
    
# ====================================
# 5. 모델 성능 평가 함수
# ====================================
def evaluate_binary_model_performance(y_true_price, y_pred_price, y_true_labels, y_pred_labels):
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
# 6. 메인 실행 로직
# ====================================
unified_df = load_and_prepare_data(MARKETS)
if unified_df is not None:
    # [수정] 반환값에 feature_columns 추가
    X_seq, y_seq, y_label_seq, scaler_X, scaler_y, feature_columns, dates = create_binary_lstm_dataset(unified_df)
    
    split_idx = int(len(X_seq) * TRAIN_RATIO)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    y_label_test = y_label_seq[split_idx:]
    dates_test = dates[split_idx:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    d4le_model = build_d4le_model(input_shape, LSTM_UNITS)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("\n모델 훈련 시작...")
    history = d4le_model.fit(X_train, y_train, 
                             epochs=EPOCHS, 
                             batch_size=BATCH_SIZE, 
                             validation_data=(X_test, y_test), 
                             callbacks=[early_stopping],
                             verbose=1)
    print("모델 훈련 완료.")

    predicted_scaled = d4le_model.predict(X_test)
    predicted_prices = scaler_y.inverse_transform(predicted_scaled)
    actual_prices = scaler_y.inverse_transform(y_test)

    last_prices_test = unified_df['usdt_close'].loc[dates_test - pd.Timedelta(days=1)].values
    predicted_returns = (predicted_prices.flatten() - last_prices_test) / last_prices_test * 100
    
    y_pred_labels = np.zeros_like(predicted_returns, dtype=int)
    y_pred_labels[predicted_returns > 0] = 1

    cm = evaluate_binary_model_performance(actual_prices, predicted_prices, y_label_test, y_pred_labels)
    
    # 시각화
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(15, 7))
    plt.plot(dates_test, actual_prices, color='cyan', label='Original Price')
    plt.plot(dates_test, predicted_prices, color='gray', label='Predicted Price')
    plt.title('D4LE Model: Actual vs. Predicted Price (Accuracy Enhanced)')
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

    # 내일 방향성 예측
    print("\n" + "="*50)
    print("내일 ETH 가격 방향성 예측")
    print("="*50)
    
    # [수정] 훈련 시 사용했던 feature_columns를 그대로 사용하여 데이터 준비
    last_sequence_data = unified_df.iloc[-LOOK_BACK:][feature_columns]
    last_sequence_scaled = scaler_X.transform(last_sequence_data).reshape(1, LOOK_BACK, last_sequence_data.shape[1])
    
    tomorrow_pred_scaled = d4le_model.predict(last_sequence_scaled)
    tomorrow_pred_price = scaler_y.inverse_transform(tomorrow_pred_scaled)[0][0]
    
    last_actual_price = unified_df['usdt_close'].iloc[-1]
    predicted_return = (tomorrow_pred_price - last_actual_price) / last_actual_price * 100
    
    direction = "SHORT (하락)" if predicted_return < 0 else "LONG (상승)"
    confidence_proxy = abs(predicted_return)

    print(f"오늘 종가 (USDT): {last_actual_price:,.2f}")
    print(f"내일 예측 종가 (USDT): {tomorrow_pred_price:,.2f}")
    print(f"예상 수익률: {predicted_return:.2f}%")
    print("-" * 50)
    print(f"💡 최종 예측 방향: {direction}")
    print(f"(예상 변동폭 기반 신뢰도: {confidence_proxy:.2f}%)")
    print("=" * 50)
