"""
Ethereum Price Prediction - Data Loading & Preprocessing
"""
# ============================================================================
# 기본 라이브러리 및 유틸리티
# ============================================================================
import gc
import json
import joblib
import os
import warnings
from dotenv import load_dotenv
# 날짜/시간
from datetime import datetime, timedelta

# 데이터 처리
import numpy as np
import pandas as pd
import pandas_ta as ta
from collections import Counter
from numba import jit
# ============================================================================
# ML/DL 라이브러리 및 도구
# ============================================================================

# 하이퍼파라미터 최적화
import optuna

# Scikit-learn: 데이터 전처리
from sklearn.feature_selection import (
    SelectKBest, RFE,
    mutual_info_classif, mutual_info_regression
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler

# Scikit-learn: 모델
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Scikit-learn: 앙상블 모델
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, # 추가된 항목
    RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor
)

# Scikit-learn: 평가 지표
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, # 분류 지표
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error # 회귀 지표
)


from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm.callback import early_stopping 
from xgboost import XGBClassifier, XGBRegressor

# TensorFlow/Keras 딥러닝
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    # 기본 레이어
    Input, Dense, Flatten, Dropout, Activation,
    # RNN 레이어
    LSTM, GRU, SimpleRNN, Bidirectional,
    # CNN 레이어
    Conv1D, MaxPooling1D, AveragePooling1D,
    GlobalAveragePooling1D, GlobalMaxPooling1D,
    # 정규화 레이어
    BatchNormalization, LayerNormalization,
    # Attention 레이어
    Attention, MultiHeadAttention,
    # 유틸리티 레이어
    Concatenate, Add, Multiply, Lambda,
    Reshape, Permute, RepeatVector, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 시계열 분석 (Statsmodels)
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

# PyTorch 
try:
    import torch
    import torch.nn as nn
except ImportError:
    pass




import optuna
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier,
    VotingClassifier, HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# ============================================================================
# 환경 설정 및 경고 무시
# ============================================================================

# GPU 메모리 증가 허용 설정
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

warnings.filterwarnings('ignore')


DATA_DIR_MAIN = './macro_data'
DATA_DIR_NEW = './macro_data/macro_data'

TRAIN_START_DATE = pd.to_datetime('2020-01-01')
LOOKBACK_DAYS = 200
LOOKBACK_START_DATE = TRAIN_START_DATE - timedelta(days=LOOKBACK_DAYS)


def standardize_date_column(df,file_name):
    """날짜 컬럼 자동 탐지 + datetime 통일 + tz 제거 + 시각 제거"""

    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if not date_cols:
        print("[Warning] 날짜 컬럼을 찾을 수 없습니다.")
        return df
    date_col = date_cols[0]
    
    if date_col != 'date':
        df.rename(columns={date_col: 'date'}, inplace=True)
    

    if file_name == 'eth_onchain.csv':
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
    
    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.normalize()  
    if pd.api.types.is_datetime64tz_dtype(df['date']):
        df['date'] = df['date'].dt.tz_convert(None)
    else:
        df['date'] = df['date'].dt.tz_localize(None)
    print(df.shape,file_name)
    return df


def load_csv(directory, filename):
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print(f"[Warning] {filename} not found")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    return standardize_date_column(df, filename)


def add_prefix(df, prefix):
    if df.empty:
        return df
    df.columns = [f"{prefix}_{col}" if col != 'date' else col for col in df.columns]
    return df


def create_sentiment_features(news_df):
    if news_df.empty:
        return pd.DataFrame(columns=['date'])
    
    agg = news_df.groupby('date').agg(
        sentiment_mean=('label', 'mean'),
        sentiment_std=('label', 'std'),
        news_count=('label', 'count'),
        positive_ratio=('label', lambda x: (x == 1).sum() / len(x)),
        negative_ratio=('label', lambda x: (x == -1).sum() / len(x)),
        extreme_positive_count=('label', lambda x: (x == 1).sum()),
        extreme_negative_count=('label', lambda x: (x == -1).sum()),
        sentiment_sum=('label', 'sum'),
    ).reset_index().fillna(0)
    
    agg['sentiment_polarity'] = agg['positive_ratio'] - agg['negative_ratio']
    agg['sentiment_intensity'] = agg['positive_ratio'] + agg['negative_ratio']
    agg['sentiment_disagreement'] = agg['positive_ratio'] * agg['negative_ratio']
    agg['bull_bear_ratio'] = agg['positive_ratio'] / (agg['negative_ratio'] + 1e-10)
    agg['weighted_sentiment'] = agg['sentiment_mean'] * np.log1p(agg['news_count'])
    agg['extremity_index'] = (agg['extreme_positive_count'] + agg['extreme_negative_count']) / (agg['news_count'] + 1e-10)
    
    for window in [3,7]:
        agg[f'sentiment_ma{window}'] = agg['sentiment_mean'].rolling(window=window, min_periods=1).mean()
        agg[f'sentiment_volatility_{window}'] = agg['sentiment_mean'].rolling(window=window, min_periods=1).std()
    
    agg['sentiment_trend'] = agg['sentiment_mean'].diff()
    agg['sentiment_acceleration'] = agg['sentiment_trend'].diff()
    agg['news_volume_change'] = agg['news_count'].pct_change()
    
    for window in [7, 14]:
        agg[f'news_volume_ma{window}'] = agg['news_count'].rolling(window=window, min_periods=1).mean()
    
    return agg.fillna(0)


def smart_fill_missing(df_merged):
    REFERENCE_START_DATE = pd.to_datetime('2020-01-01')
    
    for col in df_merged.columns:
        if col == 'date':
            continue
        
        if df_merged[col].isnull().sum() == 0:
            continue
        
        non_null_idx = df_merged[col].first_valid_index()
        
        if non_null_idx is None:
            df_merged[col] = df_merged[col].fillna(0)
            continue
        
        first_date = df_merged.loc[non_null_idx, 'date']
        
        before_mask = df_merged['date'] < first_date
        after_mask = df_merged['date'] >= first_date
        
        df_merged.loc[before_mask, col] = df_merged.loc[before_mask, col].fillna(0)
        df_merged.loc[after_mask, col] = df_merged.loc[after_mask, col].fillna(method='ffill')
        
        remaining = df_merged.loc[after_mask, col].isnull().sum()
        if remaining > 0:
            df_merged.loc[after_mask, col] = df_merged.loc[after_mask, col].fillna(0)
    
    return df_merged


print("="*80)
print("DATA LOADING")
print("="*80)

#news_df = load_csv(DATA_DIR_MAIN, 'news_data.csv')
eth_onchain_df = load_csv(DATA_DIR_MAIN, 'eth_onchain.csv')
macro_df = load_csv(DATA_DIR_NEW, 'macro_crypto_data.csv')
sp500_df = load_csv(DATA_DIR_NEW, 'SP500.csv')
vix_df = load_csv(DATA_DIR_NEW, 'VIX.csv')
gold_df = load_csv(DATA_DIR_NEW, 'GOLD.csv')
dxy_df = load_csv(DATA_DIR_NEW, 'DXY.csv')
fear_greed_df = load_csv(DATA_DIR_NEW, 'fear_greed.csv')
eth_funding_df = load_csv(DATA_DIR_NEW, 'eth_funding_rate.csv')
usdt_eth_mcap_df = load_csv(DATA_DIR_NEW, 'usdt_eth_mcap.csv')
aave_tvl_df = load_csv(DATA_DIR_NEW, 'aave_eth_tvl.csv')
lido_tvl_df = load_csv(DATA_DIR_NEW, 'lido_eth_tvl.csv')
makerdao_tvl_df = load_csv(DATA_DIR_NEW, 'makerdao_eth_tvl.csv')
uniswap_tvl_df = load_csv(DATA_DIR_NEW, 'uniswap_eth_tvl.csv')
curve_tvl_df = load_csv(DATA_DIR_NEW, 'curve-dex_eth_tvl.csv')
eth_chain_tvl_df = load_csv(DATA_DIR_NEW, 'eth_chain_tvl.csv')
layer2_tvl_df = load_csv(DATA_DIR_NEW, 'layer2_tvl.csv')

print(f"Loaded {len([df for df in [fear_greed_df, eth_funding_df, usdt_eth_mcap_df, aave_tvl_df, lido_tvl_df, makerdao_tvl_df, uniswap_tvl_df, curve_tvl_df, eth_chain_tvl_df, layer2_tvl_df] if not df.empty])} files")

all_dataframes = [
    macro_df, eth_onchain_df, fear_greed_df, usdt_eth_mcap_df,
    aave_tvl_df, lido_tvl_df, makerdao_tvl_df, uniswap_tvl_df, curve_tvl_df,
    eth_chain_tvl_df, eth_funding_df, layer2_tvl_df, 
    sp500_df, vix_df, gold_df, dxy_df#,news_df,
]

last_dates = [
    pd.to_datetime(df['date']).max() 
    for df in all_dataframes 
    if not df.empty and 'date' in df.columns
]

end_date = min(last_dates) if last_dates else pd.Timestamp.today()

print("\n" + "="*80)
print("SENTIMENT FEATURES")
print("="*80)
#sentiment_features = create_sentiment_features(news_df)
#print(f"Generated {sentiment_features.shape[1]-1} features")

print("\n" + "="*80)
print("DATA MERGING")
print("="*80)

eth_onchain_df = add_prefix(eth_onchain_df, 'eth')
fear_greed_df = add_prefix(fear_greed_df, 'fg')
usdt_eth_mcap_df = add_prefix(usdt_eth_mcap_df, 'usdt')
aave_tvl_df = add_prefix(aave_tvl_df, 'aave')
lido_tvl_df = add_prefix(lido_tvl_df, 'lido')
makerdao_tvl_df = add_prefix(makerdao_tvl_df, 'makerdao')
uniswap_tvl_df = add_prefix(uniswap_tvl_df, 'uniswap')
curve_tvl_df = add_prefix(curve_tvl_df, 'curve')
eth_chain_tvl_df = add_prefix(eth_chain_tvl_df, 'chain')
eth_funding_df = add_prefix(eth_funding_df, 'funding')
layer2_tvl_df = add_prefix(layer2_tvl_df, 'l2')
sp500_df = add_prefix(sp500_df, 'sp500')
vix_df = add_prefix(vix_df, 'vix')
gold_df = add_prefix(gold_df, 'gold')
dxy_df = add_prefix(dxy_df, 'dxy')

date_range = pd.date_range(start=LOOKBACK_START_DATE, end=end_date, freq='D')
df_merged = pd.DataFrame(date_range, columns=['date'])

dataframes_to_merge = [
    macro_df, eth_onchain_df, fear_greed_df, usdt_eth_mcap_df,
    aave_tvl_df, lido_tvl_df, makerdao_tvl_df, uniswap_tvl_df, curve_tvl_df,
    eth_chain_tvl_df, eth_funding_df, layer2_tvl_df,
    sp500_df, vix_df, gold_df, dxy_df#,sentiment_features,
]

for df in dataframes_to_merge:
    if not df.empty:
        df_merged = pd.merge(df_merged, df, on='date', how='left')

print(f"Merged shape: {df_merged.shape}")
print(f"Missing before fill: {df_merged.isnull().sum().sum():,}")

print("\n" + "="*80)
print("MISSING VALUE HANDLING")
print("="*80)

df_merged = smart_fill_missing(df_merged)

missing_after = df_merged.isnull().sum().sum()
print(f"Missing after fill: {missing_after:,}")

if missing_after > 0:
    df_merged = df_merged.fillna(0)
    print(f"Remaining filled with 0")

lookback_df = df_merged[df_merged['date'] < TRAIN_START_DATE]
cols_to_drop = [
    col for col in lookback_df.columns 
    if lookback_df[col].isnull().all() and col != 'date'
]

if cols_to_drop:
    print(f"\nDropping {len(cols_to_drop)} fully missing columns")
    df_merged = df_merged.drop(columns=cols_to_drop)

print(f"Shape: {df_merged.shape}")
print(f"Period: {df_merged['date'].min().date()} ~ {df_merged['date'].max().date()}")
print(f"Missing: {df_merged.isnull().sum().sum()}")

print(f"\nFeature groups:")
print(f"  Crypto prices: {len([c for c in df_merged.columns if any(x in c for x in ['BTC_', 'ETH_', 'BNB_', 'XRP_', 'SOL_', 'ADA_', 'DOGE_', 'AVAX_', 'DOT_'])])}")
print(f"  On-chain: {len([c for c in df_merged.columns if c.startswith('eth_')])}")
print(f"  DeFi TVL: {len([c for c in df_merged.columns if any(x in c for x in ['aave_', 'lido_', 'makerdao_', 'uniswap_', 'curve_', 'chain_'])])}")
print(f"  Layer 2: {len([c for c in df_merged.columns if c.startswith('l2_')])}")
print(f"  Sentiment: {len([c for c in df_merged.columns if any(x in c for x in ['sentiment', 'news', 'bull_bear', 'positive', 'negative', 'extreme'])])}")
print(f"  Macro: {len([c for c in df_merged.columns if any(x in c for x in ['sp500_', 'vix_', 'gold_', 'dxy_'])])}")
print(f"  Fear & Greed: {len([c for c in df_merged.columns if c.startswith('fg_')])}")
print(f"  Funding Rate: {len([c for c in df_merged.columns if c.startswith('funding_')])}")
print(f"  Stablecoin: {len([c for c in df_merged.columns if c.startswith('usdt_')])}")
print("="*80)
