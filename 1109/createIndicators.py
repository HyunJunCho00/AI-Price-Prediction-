def add_indicator_to_df(df_ta, indicator):
    """pandas_ta 지표 결과를 DataFrame에 안전하게 추가"""
    if indicator is None:
        return

    if isinstance(indicator, pd.DataFrame) and not indicator.empty:
        for col in indicator.columns:
            df_ta[col] = indicator[col]
    elif isinstance(indicator, pd.Series) and not indicator.empty:
        colname = indicator.name if indicator.name else 'Unnamed'
        df_ta[colname] = indicator

def safe_add(df_ta, func, *args, **kwargs):
    """지표 생성 시 오류 방지를 위한 래퍼 함수"""
    try:
        result = func(*args, **kwargs)
        add_indicator_to_df(df_ta, result)
        return True
    except Exception as e:
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        print(f"    ⚠ {func_name.upper()} 생성 실패: {str(e)[:50]}")
        return False

def calculate_technical_indicators(df):
    df = df.sort_values('date').reset_index(drop=True)
    df_ta = df.copy()

    close = df['ETH_Close']
    high = df.get('ETH_High', close)
    low = df.get('ETH_Low', close)
    volume = df.get('ETH_Volume', pd.Series(index=df.index, data=1))
    open_ = df.get('ETH_Open', close)

    try:
        # ===== MOMENTUM INDICATORS =====
        
        # RSI (14만 - 모든 fold 선택)
        df_ta['RSI_14'] = ta.rsi(close, length=14)
        
        # MACD (필수 - 자주 선택됨)
        safe_add(df_ta, ta.macd, close, fast=12, slow=26, signal=9)
        
        # Stochastic (14만 - 나머지는 중복)
        safe_add(df_ta, ta.stoch, high, low, close, k=14, d=3)
        
        # Williams %R
        df_ta['WILLR_14'] = ta.willr(high, low, close, length=14)
        
        # ROC (10만 - 20과 거의 동일)
        df_ta['ROC_10'] = ta.roc(close, length=10)
        
        # MOM (10만 유지)
        df_ta['MOM_10'] = ta.mom(close, length=10)
        
        # CCI (14, 50만 - 극단값 비교용)
        df_ta['CCI_14'] = ta.cci(high, low, close, length=14)
        df_ta['CCI_50'] = ta.cci(high, low, close, length=50)
        df_ta['CCI_SIGNAL'] = (df_ta['CCI_14'] > 100).astype(int)
      
        # TSI
        safe_add(df_ta, ta.tsi, close, fast=13, slow=25, signal=13)
        
        # Ichimoku (유지 - 복합 지표로 유용)
        try:
            ichimoku = ta.ichimoku(high, low, close)
            if ichimoku is not None and isinstance(ichimoku, tuple):
                ichimoku_df = ichimoku[0]
                if ichimoku_df is not None:
                    for col in ichimoku_df.columns:
                        df_ta[col] = ichimoku_df[col]
        except:
            pass

        # ===== OVERLAP INDICATORS =====
        
        # SMA (20, 50만 - Golden Cross용)
        df_ta['SMA_20'] = ta.sma(close, length=20)
        df_ta['SMA_50'] = ta.sma(close, length=50)
        
        # EMA (12, 26만 - MACD 구성 요소)
        df_ta['EMA_12'] = ta.ema(close, length=12)
        df_ta['EMA_26'] = ta.ema(close, length=26)
        
        # TEMA (10만 - 30과 중복)
        df_ta['TEMA_10'] = ta.tema(close, length=10)
        
        # WMA (20만 - 10과 중복)
        df_ta['WMA_20'] = ta.wma(close, length=20)
        
        # HMA (유지 - 독특한 smoothing)
        df_ta['HMA_9'] = ta.hma(close, length=9)
        
        # DEMA (유지)
        df_ta['DEMA_10'] = ta.dema(close, length=10)
        
        # VWMA (유지 - 거래량 가중)
        df_ta['VWMA_20'] = ta.vwma(close, volume, length=20)
        
        # 가격 조합 (유지 - 다른 정보)
        df_ta['HL2'] = ta.hl2(high, low)
        df_ta['HLC3'] = ta.hlc3(high, low, close)
        df_ta['OHLC4'] = ta.ohlc4(open_, high, low, close)

        # ===== VOLATILITY INDICATORS =====
        
        # Bollinger Bands 
        safe_add(df_ta, ta.bbands, close, length=20, std=2)
        
        # ATR 
        df_ta['ATR_14'] = ta.atr(high, low, close, length=14)
        
        # NATR
        df_ta['NATR_14'] = ta.natr(high, low, close, length=14)
        
        # True Range
        try:
            tr = ta.true_range(high, low, close)
            if isinstance(tr, pd.Series) and not tr.empty:
                df_ta['TRUERANGE'] = tr
            elif isinstance(tr, pd.DataFrame) and not tr.empty:
                df_ta['TRUERANGE'] = tr.iloc[:, 0]
        except:
            pass
        
        # Keltner Channel
        safe_add(df_ta, ta.kc, high, low, close, length=20)
        
        # Donchian Channel
        try:
            dc = ta.donchian(high, low, lower_length=20, upper_length=20)
            if dc is not None and isinstance(dc, pd.DataFrame) and not dc.empty:
                for col in dc.columns:
                    df_ta[col] = dc[col]
        except:
            pass
        
        # Supertrend
        atr_10 = ta.atr(high, low, close, length=10)
        hl2_calc = (high + low) / 2
        upper_band = hl2_calc + (3 * atr_10)
        lower_band = hl2_calc - (3 * atr_10)
        
        df_ta['SUPERTREND'] = 0
        for i in range(1, len(df_ta)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                df_ta.loc[df_ta.index[i], 'SUPERTREND'] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                df_ta.loc[df_ta.index[i], 'SUPERTREND'] = -1
            else:
                df_ta.loc[df_ta.index[i], 'SUPERTREND'] = df_ta['SUPERTREND'].iloc[i-1]

        # ===== VOLUME INDICATORS =====
        
        # OBV (필수)
        df_ta['OBV'] = ta.obv(close, volume)
        
        # AD
        df_ta['AD'] = ta.ad(high, low, close, volume)
        
        # ADOSC
        df_ta['ADOSC_3_10'] = ta.adosc(high, low, close, volume, fast=3, slow=10)
        
        # MFI
        df_ta['MFI_14'] = ta.mfi(high, low, close, volume, length=14)
        
        # CMF
        df_ta['CMF_20'] = ta.cmf(high, low, close, volume, length=20)
        
        # EFI (Fold에서 선택됨)
        df_ta['EFI_13'] = ta.efi(close, volume, length=13)
        
        # EOM
        safe_add(df_ta, ta.eom, high, low, close, volume, length=14)
        
        # VWAP
        try:
            df_ta['VWAP'] = ta.vwap(high, low, close, volume)
        except:
            pass

        # ===== TREND INDICATORS =====
        
        # ADX (필수)
        safe_add(df_ta, ta.adx, high, low, close, length=14)
        
        # Aroon
        try:
            aroon = ta.aroon(high, low, length=25)
            if aroon is not None and isinstance(aroon, pd.DataFrame):
                for col in aroon.columns:
                    df_ta[col] = aroon[col]
        except:
            pass
        
        # PSAR
        try:
            psar = ta.psar(high, low, close)
            if psar is not None:
                if isinstance(psar, pd.DataFrame) and not psar.empty:
                    for col in psar.columns:
                        df_ta[col] = psar[col]
                elif isinstance(psar, pd.Series) and not psar.empty:
                    df_ta[psar.name] = psar
        except:
            pass
        
        # Vortex 
        safe_add(df_ta, ta.vortex, high, low, close, length=14)
        
        # DPO 
        df_ta['DPO_20'] = ta.dpo(close, length=20)

        # ===== 파생 지표 =====
        
        # 가격 변화율 
        df_ta['PRICE_CHANGE'] = close.pct_change()
        
        # 변동성 
        df_ta['VOLATILITY_20'] = close.pct_change().rolling(window=20).std()
        
        # 모멘텀 
        df_ta['MOMENTUM_10'] = close / close.shift(10) - 1
        
        # 이동평균 대비 위치 
        df_ta['PRICE_VS_SMA20'] = close / df_ta['SMA_20'] - 1
        df_ta['PRICE_VS_EMA12'] = close / df_ta['EMA_12'] - 1
        
        # 크로스 신호 
        df_ta['SMA_GOLDEN_CROSS'] = (df_ta['SMA_50'] > df_ta['SMA_20']).astype(int)
        df_ta['EMA_CROSS_SIGNAL'] = (df_ta['EMA_12'] > df_ta['EMA_26']).astype(int)
        
        # 거래량 지표
        df_ta['VOLUME_SMA_20'] = ta.sma(volume, length=20)
        df_ta['VOLUME_RATIO'] = volume / (df_ta['VOLUME_SMA_20'] + 1e-10)
        df_ta['VOLUME_CHANGE'] = volume.pct_change()
        df_ta['VOLUME_CHANGE_5'] = volume.pct_change(periods=5)
        
        # Range 지표 
        df_ta['HIGH_LOW_RANGE'] = (high - low) / (close + 1e-10)
        df_ta['HIGH_CLOSE_RANGE'] = np.abs(high - close.shift()) / (close + 1e-10)
        df_ta['CLOSE_LOW_RANGE'] = (close - low) / (close + 1e-10)
        
        # 일중 가격 위치
        df_ta['INTRADAY_POSITION'] = (close - low) / ((high - low) + 1e-10)
        
        # Linear Regression Slope 
        try:
            df_ta['SLOPE_5'] = ta.linreg(close, length=5, slope=True)
        except:
            df_ta['SLOPE_5'] = close.rolling(window=5).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 5 else np.nan, raw=True
            )
        
        # Increasing 
        df_ta['INC_1'] = (close > close.shift(1)).astype(int)
        
        # BOP
        df_ta['BOP'] = (close - open_) / ((high - low) + 1e-10)
        df_ta['BOP'] = df_ta['BOP'].fillna(0)
        
        # ===== 고급 파생 지표 =====
        
        # Bollinger Bands 파생 
        if 'BBL_20' in df_ta.columns and 'BBU_20' in df_ta.columns and 'BBM_20' in df_ta.columns:
            df_ta['BB_WIDTH'] = (df_ta['BBU_20'] - df_ta['BBL_20']) / (df_ta['BBM_20'] + 1e-8)
            df_ta['BB_POSITION'] = (close - df_ta['BBL_20']) / (df_ta['BBU_20'] - df_ta['BBL_20'] + 1e-8)
        
        # RSI 파생
        df_ta['RSI_OVERBOUGHT'] = (df_ta['RSI_14'] > 70).astype(int)
        df_ta['RSI_OVERSOLD'] = (df_ta['RSI_14'] < 30).astype(int)
        
        # MACD 히스토그램 변화율
        if 'MACDh_12_26_9' in df_ta.columns:
            df_ta['MACD_HIST_CHANGE'] = df_ta['MACDh_12_26_9'].diff()
        
        # Volume Profile
        df_ta['VOLUME_STRENGTH'] = volume / volume.rolling(window=50).mean()
        
        # Price Acceleration
        df_ta['PRICE_ACCELERATION'] = close.pct_change().diff()
        
        # Gap (Fold에서 선택됨)
        df_ta['GAP'] = (open_ - close.shift(1)) / (close.shift(1) + 1e-10)
        
        # Distance from High/Low 
        df_ta['ROLLING_MAX_20'] = close.rolling(window=20).max()
        df_ta['ROLLING_MIN_20'] = close.rolling(window=20).min()
        df_ta['DISTANCE_FROM_HIGH'] = (df_ta['ROLLING_MAX_20'] - close) / (df_ta['ROLLING_MAX_20'] + 1e-10)
        df_ta['DISTANCE_FROM_LOW'] = (close - df_ta['ROLLING_MIN_20']) / (close + 1e-10)

        # Realized Volatility 
        ret_squared = close.pct_change() ** 2
        df_ta['RV_5'] = ret_squared.rolling(5).sum()
        df_ta['RV_20'] = ret_squared.rolling(20).sum()
        df_ta['RV_RATIO'] = df_ta['RV_5'] / (df_ta['RV_20'] + 1e-10)
        
        added = df_ta.shape[1] - df.shape[1]

    except Exception as e:
        print(f"\n❌ Error: {e}")

    return df_ta



def add_enhanced_cross_crypto_features(df):
    df_enhanced = df.copy()
    df_enhanced['eth_return'] = df['ETH_Close'].pct_change()
    df_enhanced['btc_return'] = df['BTC_Close'].pct_change()

    for lag in [1, 5]:
        df_enhanced[f'btc_return_lag{lag}'] = df_enhanced['btc_return'].shift(lag)

    for window in [3, 7, 14, 30, 60]:
        df_enhanced[f'eth_btc_corr_{window}d'] = (
            df_enhanced['eth_return'].rolling(window).corr(df_enhanced['btc_return'])
        )

    eth_vol = df_enhanced['eth_return'].abs()
    btc_vol = df_enhanced['btc_return'].abs()

    for window in [7, 14, 30]:
        df_enhanced[f'eth_btc_volcorr_{window}d'] = eth_vol.rolling(window).corr(btc_vol)
        df_enhanced[f'eth_btc_volcorr_sq_{window}d'] = (
            (df_enhanced['eth_return']**2).rolling(window).corr(df_enhanced['btc_return']**2)
        )

    df_enhanced['btc_eth_strength_ratio'] = (
        df_enhanced['btc_return'] / (df_enhanced['eth_return'].abs() + 1e-8)
    )
    df_enhanced['btc_eth_strength_ratio_7d'] = df_enhanced['btc_eth_strength_ratio'].rolling(7).mean()

    alt_returns = []
    for coin in ['BNB', 'XRP', 'SOL', 'ADA']:
        if f'{coin}_Close' in df.columns:
            alt_returns.append(df[f'{coin}_Close'].pct_change())

    if alt_returns:
        market_return = pd.concat(
            alt_returns + [df_enhanced['eth_return'], df_enhanced['btc_return']], axis=1
        ).mean(axis=1)
        df_enhanced['btc_dominance'] = df_enhanced['btc_return'] / (market_return + 1e-8)

    for window in [30, 60, 90]:
        covariance = df_enhanced['eth_return'].rolling(window).cov(df_enhanced['btc_return'])
        btc_variance = df_enhanced['btc_return'].rolling(window).var()
        df_enhanced[f'eth_btc_beta_{window}d'] = covariance / (btc_variance + 1e-8)

    df_enhanced['eth_btc_spread'] = df_enhanced['eth_return'] - df_enhanced['btc_return']
    df_enhanced['eth_btc_spread_ma7'] = df_enhanced['eth_btc_spread'].rolling(7).mean()
    df_enhanced['eth_btc_spread_std7'] = df_enhanced['eth_btc_spread'].rolling(7).std()

    btc_vol_ma = btc_vol.rolling(30).mean()
    high_vol_mask = btc_vol > btc_vol_ma
    df_enhanced['eth_btc_corr_highvol'] = np.nan
    df_enhanced['eth_btc_corr_lowvol'] = np.nan

    for i in range(30, len(df_enhanced)):
        window_data = df_enhanced.iloc[i-30:i]
        high_vol_data = window_data[high_vol_mask.iloc[i-30:i]]
        low_vol_data = window_data[~high_vol_mask.iloc[i-30:i]]

        if len(high_vol_data) > 5:
            df_enhanced.loc[df_enhanced.index[i], 'eth_btc_corr_highvol'] = (
                high_vol_data['eth_return'].corr(high_vol_data['btc_return'])
            )
        if len(low_vol_data) > 5:
            df_enhanced.loc[df_enhanced.index[i], 'eth_btc_corr_lowvol'] = (
                low_vol_data['eth_return'].corr(low_vol_data['btc_return'])
            )

    return df_enhanced


def remove_raw_prices_and_transform(df):
    df_transformed = df.copy()

    if 'eth_log_return' not in df_transformed.columns:
        df_transformed['eth_log_return'] = np.log(df['ETH_Close'] / df['ETH_Close'].shift(1))
    if 'eth_intraday_range' not in df_transformed.columns:
        df_transformed['eth_intraday_range'] = (df['ETH_High'] - df['ETH_Low']) / (df['ETH_Close'] + 1e-8)
    if 'eth_body_ratio' not in df_transformed.columns:
        df_transformed['eth_body_ratio'] = (df['ETH_Close'] - df['ETH_Open']) / (df['ETH_Close'] + 1e-8)
    if 'eth_close_position' not in df_transformed.columns:
        df_transformed['eth_close_position'] = (
            (df['ETH_Close'] - df['ETH_Low']) / (df['ETH_High'] - df['ETH_Low'] + 1e-8)
        )

    if 'BTC_Close' in df_transformed.columns:
        for period in [5, 20]:
            col_name = f'btc_return_{period}d'
            if col_name not in df_transformed.columns:
                df_transformed[col_name] = np.log(df['BTC_Close'] / df['BTC_Close'].shift(period)).fillna(0)
        
        for period in [7, 14, 30]:
            col_name = f'btc_volatility_{period}d'
            if col_name not in df_transformed.columns:
                df_transformed[col_name] = (
                    df_transformed['eth_log_return'].rolling(period, min_periods=max(3, period//3)).std()
                ).fillna(0)
        
        if 'btc_intraday_range' not in df_transformed.columns:
            df_transformed['btc_intraday_range'] = (df['BTC_High'] - df['BTC_Low']) / (df['BTC_Close'] + 1e-8)
        if 'btc_body_ratio' not in df_transformed.columns:
            df_transformed['btc_body_ratio'] = (df['BTC_Close'] - df['BTC_Open']) / (df['BTC_Close'] + 1e-8)

        if 'BTC_Volume' in df.columns:
            btc_volume = df['BTC_Volume']
            if 'btc_volume_change' not in df_transformed.columns:
                df_transformed['btc_volume_change'] = btc_volume.pct_change().fillna(0)
            if 'btc_volume_ratio_20d' not in df_transformed.columns:
                volume_ma20 = btc_volume.rolling(20, min_periods=5).mean()
                df_transformed['btc_volume_ratio_20d'] = (btc_volume / (volume_ma20 + 1e-8)).fillna(1)
            if 'btc_volume_volatility_30d' not in df_transformed.columns:
                df_transformed['btc_volume_volatility_30d'] = (
                    btc_volume.pct_change().rolling(30, min_periods=10).std()
                ).fillna(0)
            if 'btc_obv' not in df_transformed.columns:
                btc_close = df['BTC_Close']
                obv = np.where(btc_close > btc_close.shift(1), btc_volume,
                               np.where(btc_close < btc_close.shift(1), -btc_volume, 0))
                df_transformed['btc_obv'] = pd.Series(obv, index=df.index).cumsum().fillna(0)
            if 'btc_volume_price_corr_30d' not in df_transformed.columns:
                df_transformed['btc_volume_price_corr_30d'] = (
                    btc_volume.pct_change().rolling(30, min_periods=10).corr(
                        df_transformed['eth_log_return']
                    )
                ).fillna(0)

    altcoins = ['BNB', 'XRP', 'SOL', 'ADA', 'DOGE', 'AVAX', 'DOT']
    for coin in altcoins:
        if f'{coin}_Close' in df_transformed.columns:
            col_name = f'{coin.lower()}_return'
            if col_name not in df_transformed.columns:
                df_transformed[col_name] = np.log(df[f'{coin}_Close'] / df[f'{coin}_Close'].shift(1)).fillna(0)
            vol_col = f'{coin.lower()}_volatility_30d'
            if vol_col not in df_transformed.columns:
                df_transformed[vol_col] = df_transformed[col_name].rolling(30, min_periods=10).std().fillna(0)
            
            if f'{coin}_Volume' in df.columns:
                coin_volume = df[f'{coin}_Volume']
                volume_change_col = f'{coin.lower()}_volume_change'
                if volume_change_col not in df_transformed.columns:
                    df_transformed[volume_change_col] = coin_volume.pct_change().fillna(0)
                volume_ratio_col = f'{coin.lower()}_volume_ratio_20d'
                if volume_ratio_col not in df_transformed.columns:
                    volume_ma20 = coin_volume.rolling(20, min_periods=5).mean()
                    df_transformed[volume_ratio_col] = (coin_volume / (volume_ma20 + 1e-8)).fillna(1)

    if 'ETH_Volume' in df.columns and 'BTC_Volume' in df.columns:
        eth_volume = df['ETH_Volume']
        btc_volume = df['BTC_Volume']
        if 'eth_btc_volume_corr_30d' not in df_transformed.columns:
            df_transformed['eth_btc_volume_corr_30d'] = (
                eth_volume.pct_change().rolling(30, min_periods=10).corr(btc_volume.pct_change())
            ).fillna(0)
        if 'eth_btc_volume_ratio' not in df_transformed.columns:
            df_transformed['eth_btc_volume_ratio'] = (eth_volume / (btc_volume + 1e-8)).fillna(0)
        if 'eth_btc_volume_ratio_ma30' not in df_transformed.columns:
            df_transformed['eth_btc_volume_ratio_ma30'] = (
                df_transformed['eth_btc_volume_ratio'].rolling(30, min_periods=10).mean()
            ).fillna(0)

    remove_patterns = ['_Close', '_Open', '_High', '_Low', '_Volume']
    cols_to_remove = [
        col for col in df_transformed.columns
        if any(p in col for p in remove_patterns)
        and not any(d in col.lower() for d in ['_lag', '_position', '_ratio', '_range', '_change', '_corr', '_volatility', '_obv'])
    ]
    df_transformed.drop(cols_to_remove, axis=1, inplace=True)

    return_cols = [col for col in df_transformed.columns if 'return' in col.lower() and 'next' not in col]
    if return_cols:
        df_transformed[return_cols] = df_transformed[return_cols].fillna(0)

    return df_transformed

def apply_lag_features(df, news_lag=2, onchain_lag=1):
    df_lagged = df.copy()
    
    raw_sentiment_cols = ['sentiment_mean', 'sentiment_std', 'news_count', 'positive_ratio', 'negative_ratio']
    sentiment_ma_cols = [col for col in df.columns if 'sentiment' in col and ('_ma7' in col or '_volatility_7' in col)]
    no_lag_patterns = ['_trend', '_acceleration', '_volume_change', 'news_volume_change', 'news_volume_ma']
    onchain_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['eth_tx', 'eth_active', 'eth_new', 'eth_large', 'eth_token', 
                                  'eth_contract', 'eth_avg_gas', 'eth_total_gas', 'eth_avg_block'])]
    other_cols = [col for col in df.columns if any(keyword in col.lower() 
                  for keyword in ['tvl', 'funding', 'lido_', 'aave_', 'makerdao_', 
                                'chain_', 'usdt_', 'sp500_', 'vix_', 'gold_', 'dxy_', 'fg_'])]
    
    exclude_cols = ['ETH_Close', 'ETH_High', 'ETH_Low', 'ETH_Open', 'date']
    exclude_cols.extend([col for col in df.columns if 'event_' in col or 'period_' in col or '_lag' in col])
    
    cols_to_drop = []
    
    for col in raw_sentiment_cols:
        if col in df.columns:
            for lag in range(1, news_lag + 1):
                df_lagged[f"{col}_lag{lag}"] = df[col].shift(lag)
            cols_to_drop.append(col)
    
    for col in sentiment_ma_cols:
        if col in df.columns and col not in cols_to_drop:
            if not any(pattern in col for pattern in no_lag_patterns):
                df_lagged[f"{col}_lag1"] = df[col].shift(1)
                cols_to_drop.append(col)
    
    for col in onchain_cols:
        if col not in exclude_cols:
            df_lagged[f"{col}_lag1"] = df[col].shift(onchain_lag)
            if col in df.columns:
                cols_to_drop.append(col)
    
    for col in other_cols:
        if col not in exclude_cols:
            df_lagged[f"{col}_lag1"] = df[col].shift(1)
            if col in df.columns:
                cols_to_drop.append(col)
    
    df_lagged.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df_lagged


def add_price_lag_features_first(df):
    df_new = df.copy()
    close = df['ETH_Close']
    high = df['ETH_High']
    low = df['ETH_Low']
    volume = df['ETH_Volume']
    
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        df_new[f'close_lag{lag}'] = close.shift(lag)
    
    for lag in [1, 2, 3, 5, 7]:
        df_new[f'high_lag{lag}'] = high.shift(lag)
        df_new[f'low_lag{lag}'] = low.shift(lag)
        df_new[f'volume_lag{lag}'] = volume.shift(lag)
        df_new[f'return_lag{lag}'] = close.pct_change(periods=lag).shift(1)
    
    for lag in [1, 7, 30]:
        df_new[f'close_ratio_lag{lag}'] = close / close.shift(lag)
    
    return df_new

def add_interaction_features(df):
    df_interact = df.copy()
    
    if 'RSI_14' in df.columns and 'VOLUME_RATIO' in df.columns:
        df_interact['RSI_Volume_Strength'] = df['RSI_14'] * df['VOLUME_RATIO']
    
    if 'vix_VIX' in df.columns and 'VOLATILITY_20' in df.columns:
        df_interact['VIX_ETH_Vol_Cross'] = df['vix_VIX'] * df['VOLATILITY_20']
    
    if 'MACD_12_26_9' in df.columns and 'VOLUME_RATIO' in df.columns:
        df_interact['MACD_Volume_Momentum'] = df['MACD_12_26_9'] * df['VOLUME_RATIO']
    
    if 'btc_return' in df.columns and 'eth_btc_corr_30d' in df.columns:
        df_interact['BTC_Weighted_Impact'] = df['btc_return'] * df['eth_btc_corr_30d']
    
    if 'ATR_14' in df.columns and 'VOLUME_RATIO' in df.columns:
        df_interact['Liquidity_Risk'] = df['ATR_14'] * (1 / (df['VOLUME_RATIO'] + 1e-8))
    
    return df_interact

def add_volatility_regime_features(df):
    df_regime = df.copy()
    
    if 'VOLATILITY_20' in df.columns:
        vol_median = df['VOLATILITY_20'].rolling(60, min_periods=20).median()
        df_regime['vol_regime_high'] = (df['VOLATILITY_20'] > vol_median).astype(int)
        
        vol_mean = df['VOLATILITY_20'].rolling(30, min_periods=10).mean()
        vol_std = df['VOLATILITY_20'].rolling(30, min_periods=10).std()
        df_regime['vol_spike'] = (df['VOLATILITY_20'] > vol_mean + 2 * vol_std).astype(int)
        
        df_regime['vol_percentile_90d'] = df['VOLATILITY_20'].rolling(90, min_periods=30).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
        )
        df_regime['vol_trend'] = df['VOLATILITY_20'].pct_change(5)
        df_regime['vol_regime_duration'] = df_regime.groupby(
            (df_regime['vol_regime_high'] != df_regime['vol_regime_high'].shift()).cumsum()
        ).cumcount() + 1

    return df_regime


def add_normalized_price_lags(df):
    df_norm = df.copy()
    
    if 'ETH_Close' not in df.columns:
        return df_norm
    
    current_close = df['ETH_Close']
    lag_cols = [col for col in df.columns if 'close_lag' in col and col.replace('close_lag', '').isdigit()]
    
    for col in lag_cols:
        lag_num = col.replace('close_lag', '')
        df_norm[f'close_lag{lag_num}_ratio'] = df[col] / (current_close + 1e-8)
        next_lag_col = f'close_lag{int(lag_num)+1}'
        if next_lag_col in df.columns:
            df_norm[f'close_lag{lag_num}_logret'] = np.log(df[col] / (df[next_lag_col] + 1e-8))
    
    for col in df.columns:
        if 'high_lag' in col:
            lag_num = col.replace('high_lag', '')
            df_norm[f'high_lag{lag_num}_ratio'] = df[col] / (current_close + 1e-8)
        if 'low_lag' in col:
            lag_num = col.replace('low_lag', '')
            df_norm[f'low_lag{lag_num}_ratio'] = df[col] / (current_close + 1e-8)
    
    return df_norm


def add_percentile_features(df):
    df_pct = df.copy()
    
    if 'ETH_Close' in df.columns:
        df_pct['price_percentile_250d'] = df['ETH_Close'].rolling(250, min_periods=60).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
        )
    
    if 'ETH_Volume' in df.columns:
        df_pct['volume_percentile_90d'] = df['ETH_Volume'].rolling(90, min_periods=30).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
        )
    
    if 'RSI_14' in df.columns:
        df_pct['RSI_percentile_60d'] = df['RSI_14'].rolling(60, min_periods=20).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
        )
    
    return df_pct


def handle_missing_values_paper_based(df_clean, train_start_date, is_train=True, train_stats=None):
    """
    암호화폐 시계열 결측치 처리
    
    참고문헌:
    1. "Quantifying Cryptocurrency Unpredictability" (2025)

    2. "Time Series Data Forecasting" 
    
    3. "Dealing with Leaky Missing Data in Production" (2021)

    """
    
    # ===== 1. Lookback 제거 =====
    if isinstance(train_start_date, str):
        train_start_date = pd.to_datetime(train_start_date)
    
    before = len(df_clean)
    df_clean = df_clean[df_clean['date'] >= train_start_date].reset_index(drop=True)
    
    # ===== 2. Feature 컬럼 선택 =====
    target_cols = ['next_log_return', 'next_direction', 'next_close','next_open']
    feature_cols = [col for col in df_clean.columns 
                   if col not in target_cols + ['date']]
    
    # ===== 3. 결측 확인 =====
    missing_before = df_clean[feature_cols].isnull().sum().sum()
    
    # ===== 4. FFill → 0 =====
    df_clean[feature_cols] = df_clean[feature_cols].fillna(method='ffill')
    df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
    
    missing_after = df_clean[feature_cols].isnull().sum().sum()
    
    # ===== 5. 무한대 처리 =====
    inf_count = 0
    for col in feature_cols:
        if np.isinf(df_clean[col]).sum() > 0:
            inf_count += np.isinf(df_clean[col]).sum()
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(0)
    
    # ===== 6. 최종 확인 =====
    final_missing = df_clean[feature_cols].isnull().sum().sum()
    
    if final_missing > 0:
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
    
    
    if is_train:
        return df_clean, {}
    else:
        return df_clean
    
    
@jit(nopython=True)
def compute_triple_barrier_targets(
    prices_close,
    prices_high,
    prices_low,
    atr,
    lookahead_candles,
    atr_multiplier_profit,
    atr_multiplier_stop
):
    n = len(prices_close)
    targets_raw = np.zeros(n, dtype=np.int32) 

    for i in range(n - lookahead_candles):
        current_atr = max(atr[i], 1e-8) 
        current_price = prices_close[i]
        
        upper_barrier = current_price + (current_atr * atr_multiplier_profit)
        lower_barrier = current_price - (current_atr * atr_multiplier_stop)
        
        for j in range(1, lookahead_candles + 1):
            future_high = prices_high[i + j]
            future_low = prices_low[i + j]
            
            if future_high >= upper_barrier:
                targets_raw[i] = 1
                break 
                
            elif future_low <= lower_barrier:
                targets_raw[i] = 2
                break
    
    return targets_raw


def create_targets(df, lookahead_candles=8, atr_multiplier_profit=1.5, atr_multiplier_stop=1.0):
    df_target = df.copy()
    
    atr_col_name = 'ATR_14'
    if atr_col_name not in df.columns:
        raise ValueError(f"'{atr_col_name}' feature is missing. Run calculate_technical_indicators first.")

    prices_close = df_target['ETH_Close'].to_numpy()
    prices_high = df_target['ETH_High'].to_numpy()
    prices_low = df_target['ETH_Low'].to_numpy()
    atr = pd.Series(df_target[atr_col_name]).fillna(method='ffill').fillna(0).to_numpy()

    targets_raw = compute_triple_barrier_targets(
        prices_close,
        prices_high,
        prices_low,
        atr,
        lookahead_candles,
        atr_multiplier_profit,
        atr_multiplier_stop
    )
    
    next_open = df['ETH_Open'].shift(-1)
    next_close = df['ETH_Close'].shift(-1)
    
    df_target['next_log_return'] = np.log(next_close / next_open)
    
    df_target['next_direction'] = pd.Series(targets_raw, index=df_target.index).map({
        1: 1,
        2: 0,
        0: np.nan
    })
    
    df_target['next_open'] = next_open
    df_target['next_close'] = next_close
    
    return df_target



import numpy as np
import pandas as pd

def preprocess_non_stationary_features(df):
    df_proc = df.copy()
    
    prefixes_to_transform = [
        'eth_', 'aave_', 'lido_', 'makerdao_', 'uniswap_', 'curve_', 'chain_',
        'l2_', 'sp500_', 'gold_', 'dxy_', 'vix_', 'usdt_'
    ]
    
    exclude_prefixes = ['fg_', 'funding_']
    
    exclude_keywords = [
        '_pct_', '_ratio', '_lag', '_volatility', '_corr', '_beta', '_spread',
        'eth_return', 'btc_return', 'eth_log_return' 
    ]
    
    cols_to_transform = []
    for col in df_proc.columns:
        if col.startswith(tuple(prefixes_to_transform)):
            if not col.startswith(tuple(exclude_prefixes)):
                if not any(keyword in col for keyword in exclude_keywords):
                    cols_to_transform.append(col)
                    
    cols_to_drop = []

    for col in cols_to_transform:
        df_proc[col] = df_proc[col].fillna(method='ffill').replace(0, 1e-9)

        df_proc[f'{col}_pct_1d'] = df_proc[col].pct_change(1)
        df_proc[f'{col}_pct_5d'] = df_proc[col].pct_change(5)
        
        ma_30 = df_proc[col].rolling(window=30, min_periods=10).mean()
        df_proc[f'{col}_ma30_ratio'] = df_proc[col] / (ma_30 + 1e-9)
        
        cols_to_drop.append(col)

    df_proc = df_proc.drop(columns=cols_to_drop, errors='ignore')
    
    df_proc = df_proc.replace([np.inf, -np.inf], np.nan)
    df_proc = df_proc.fillna(method='ffill').fillna(0)
    
    print(f"Preprocessed and replaced {len(cols_to_drop)} non-stationary features.")
    
    return df_proc
