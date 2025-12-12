import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import datetime

# --- 1. 데이터 획득 (Data Acquisition) ---
def get_ethereum_data(start_date="2020-01-01", end_date=datetime.datetime.now().strftime("%Y-%m-%d"), ticker="ETH-USD"):
    print(f"이더리움 데이터 다운로드 중: {start_date} ~ {end_date}")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("다운로드된 데이터가 없습니다.")
        df_close = data[['Close']]
        print("데이터 다운로드 완료.")
        return df_close
    except Exception as e:
        print(f"데이터 다운로드 중 오류 발생: {e}")
        return pd.DataFrame()

# --- 2. 데이터 전처리 (Data Preprocessing) ---
def preprocess_data(df, timesteps=16, train_ratio=0.7, val_ratio=0.15):
    print("\n데이터 전처리 시작...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    print("데이터 정규화 완료.")

    X, y = [], []
    for i in range(timesteps, len(scaled_data)):
        X.append(scaled_data[i-timesteps:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    print(f"슬라이딩 윈도우 (timesteps={timesteps}) 적용 완료. 생성된 시퀀스 수: {len(X)}")

    total_samples = len(X)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    print(f"학습 세트: {len(X_train)} | 검증 세트: {len(X_val)} | 테스트 세트: {len(X_test)}")

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    print(f"데이터 형태 재구성 완료. X_train.shape: {X_train.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# --- 3. CNN-BiLSTM 모델 구축 ---
def build_cnn_bilstm_model(timesteps, features=1):
    print("\nCNN-BiLSTM 모델 구축 시작...")
    model = Sequential([
        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(timesteps, features)),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(units=150, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(units=50, return_sequences=False)),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    print("모델 구축 완료.")
    model.summary()
    return model

# --- 4. 모델 훈련 ---
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=16, patience=10):
    print("\n모델 훈련 시작...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    print("모델 훈련 완료.")
    return history

# --- 5. 모델 평가 ---
def evaluate_model(model, X_test, y_test, scaler):
    print("\n모델 평가 시작...")
    y_pred_scaled = model.predict(X_test)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_predicted = scaler.inverse_transform(y_pred_scaled)
    mape = mean_absolute_percentage_error(y_actual, y_predicted) * 100
    r2 = r2_score(y_actual, y_predicted)
    print(f"MAPE: {mape:.4f}% | R²: {r2:.4f}")
    print("모델 평가 완료.")
    return y_actual, y_predicted

# --- 결과 시각화 ---
def plot_results(y_actual, y_predicted):
    plt.figure(figsize=(14, 7))
    plt.plot(y_actual, label='Actual Price', color='blue')
    plt.plot(y_predicted, label='Predicted Price', color='red', linestyle='--')
    plt.title("Ethereum Price Prediction: Actual vs Predicted")
    plt.xlabel('Time (Test Samples)')
    plt.ylabel('ETH Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    TIMESTEPS = 16

    # 1. 데이터 획득
    df_ethereum = get_ethereum_data()
    if not df_ethereum.empty:
        # 2. 데이터 전처리
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(df_ethereum, timesteps=TIMESTEPS)

        # 3. 모델 구축
        model = build_cnn_bilstm_model(timesteps=TIMESTEPS)

        # 4. 모델 훈련
        history = train_model(model, X_train, y_train, X_val, y_val)
        plot_training_history(history)

        # 5. 모델 평가
        y_actual, y_predicted = evaluate_model(model, X_test, y_test, scaler)
        plot_results(y_actual, y_predicted)
        
        # --- 6. 내일 가격 최종 예측 ---
        print("\n" + "="*50)
        print("내일 이더리움 가격 예측")
        print("="*50)

        # 마지막 시퀀스 데이터 준비 (가장 최신 16일치 데이터)
        last_sequence = df_ethereum['Close'].values[-TIMESTEPS:]
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
        X_to_predict = last_sequence_scaled.reshape(1, TIMESTEPS, 1)

        # 내일 가격 예측
        tomorrow_price_scaled = model.predict(X_to_predict)
        tomorrow_price = scaler.inverse_transform(tomorrow_price_scaled)[0][0]

        # [수정] .values[-1]로 Numpy 배열 값을 가져온 후, .item()으로 순수 숫자 값을 추출
        today_price = df_ethereum['Close'].values[-1].item()
        percentage_change = ((tomorrow_price - today_price) / today_price) * 100

        print(f"오늘 종가 (USD): ${today_price:,.2f}")
        print(f"내일 예측 종가 (USD): ${tomorrow_price:,.2f}")
        print("-" * 50)
        print(f"💡 예상 변동률: {percentage_change:+.2f}%")
        print("=" * 50)
