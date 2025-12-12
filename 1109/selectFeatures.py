def select_features_multi_target(X_train, y_train, target_type='direction', top_n=30):
    
    atr_col_name = 'ATR_14'

    if target_type == 'direction':
        selected, stats = select_features_verified(
            X_train, 
            y_train['next_direction'], 
            task='class', 
            top_n=top_n
        )
        
        if atr_col_name not in selected and atr_col_name in X_train.columns:
            selected.pop()
            selected.insert(0, atr_col_name)
            
    elif target_type == 'return':
        selected, stats = select_features_verified(
            X_train, 
            y_train['next_log_return'], 
            task='reg', 
            top_n=top_n
        )
        
    elif target_type == 'price':
        selected, stats = select_features_verified(
            X_train, 
            y_train['next_close'], 
            task='reg', 
            top_n=top_n
        )
        
    elif target_type == 'direction_return':
        dir_features, dir_stats = select_features_verified(
            X_train, 
            y_train['next_direction'], 
            task='class', 
            top_n=top_n // 2,
            verbose=False
        )
        
        ret_features, ret_stats = select_features_verified(
            X_train, 
            y_train['next_log_return'], 
            task='reg', 
            top_n=top_n // 2,
            verbose=False
        )
        
        selected = list(dict.fromkeys(dir_features + ret_features))
        
        if len(selected) < top_n:
            all_mi_scores = {**dir_stats['mi_scores'], **ret_stats['mi_scores']}
            sorted_features = sorted(all_mi_scores.items(), key=lambda x: x[1], reverse=True)
            
            for feat, _ in sorted_features:
                if feat not in selected:
                    selected.append(feat)
                    if len(selected) >= top_n:
                        break
        
        selected = selected[:top_n]
        
        stats = {
            'dir_stats': dir_stats,
            'ret_stats': ret_stats,
            'overlap': len(set(dir_features) & set(ret_features))
        }
        
        if atr_col_name not in selected and atr_col_name in X_train.columns:
            if len(selected) == top_n:
                selected.pop()
            selected.insert(0, atr_col_name)
            
    elif target_type == 'direction_price':
        dir_features, dir_stats = select_features_verified(
            X_train, 
            y_train['next_direction'], 
            task='class', 
            top_n=top_n // 2,
            verbose=False
        )
        
        price_features, price_stats = select_features_verified(
            X_train, 
            y_train['next_close'], 
            task='reg', 
            top_n=top_n // 2,
            verbose=False
        )
        
        selected = list(dict.fromkeys(dir_features + price_features))
        
        if len(selected) < top_n:
            all_mi_scores = {**dir_stats['mi_scores'], **price_stats['mi_scores']}
            sorted_features = sorted(all_mi_scores.items(), key=lambda x: x[1], reverse=True)
            
            for feat, _ in sorted_features:
                if feat not in selected:
                    selected.append(feat)
                    if len(selected) >= top_n:
                        break
        
        selected = selected[:top_n]
        
        stats = {
            'dir_stats': dir_stats,
            'price_stats': price_stats,
            'overlap': len(set(dir_features) & set(price_features))
        }

        if atr_col_name not in selected and atr_col_name in X_train.columns:
            if len(selected) == top_n:
                selected.pop()
            selected.insert(0, atr_col_name)

    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    
    print(", ".join(selected))
    return selected, stats


def select_features_verified(X_train, y_train, task='class', top_n=30, verbose=True):
    
    if task == 'class':
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=3)
    else:
        mi_scores = mutual_info_regression(X_train, y_train, random_state=42, n_neighbors=3)
    
    mi_idx = np.argsort(mi_scores)[::-1][:top_n]
    mi_features = X_train.columns[mi_idx].tolist()
    
    if task == 'class':
        estimator = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
    else:
        estimator = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
    
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=top_n,
        step=0.1,
        verbose=0
    )
    
    rfe.fit(X_train, y_train)
    rfe_features = X_train.columns[rfe.support_].tolist()

    if task == 'class':
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    rf_model.fit(X_train, y_train)
    rf_importances = rf_model.feature_importances_
    rf_idx = np.argsort(rf_importances)[::-1][:top_n]
    rf_features = X_train.columns[rf_idx].tolist()
    
    all_features = mi_features + rfe_features + rf_features
    feature_votes = Counter(all_features)
    selected_features = [feat for feat, _ in feature_votes.most_common(top_n)]

    if len(selected_features) < top_n:
        remaining = top_n - len(selected_features)
        for feat in mi_features:
            if feat not in selected_features:
                selected_features.append(feat)
                remaining -= 1
                if remaining == 0:
                    break
    
    return selected_features, {
        'mi_features': mi_features,
        'rfe_features': rfe_features,
        'rf_features': rf_features,
        'feature_votes': feature_votes,
        'mi_scores': dict(zip(X_train.columns, mi_scores)),
        'rf_importances': dict(zip(X_train.columns, rf_importances))
    }


def split_tvt_method(df, train_start_date, test_start_date='2025-01-01', 
                     train_ratio=0.7, val_ratio=0.15):
    """
    test_start_date를 고정하고, 그 이전 데이터를 train/val로 분할
    test_start_date 이후 데이터는 모두 test로 사용
    """
    df_period = df[df['date'] >= train_start_date].copy()
    
    # 테스트 시작 날짜를 datetime으로 변환
    if isinstance(test_start_date, str):
        test_start_date = pd.to_datetime(test_start_date)
    
    # test_start_date 이전 데이터를 train/val로, 이후를 test로 분할
    pre_test_df = df_period[df_period['date'] < test_start_date].copy()
    test_df = df_period[df_period['date'] >= test_start_date].copy()
    
    # train/val 분할 (test 이전 데이터만 사용)
    n_pre_test = len(pre_test_df)
    train_end = int(n_pre_test * train_ratio / (train_ratio + val_ratio))
    
    train_df = pre_test_df.iloc[:train_end].copy()
    val_df = pre_test_df.iloc[train_end:].copy()
    
    print(f"\n{'='*80}")
    print(f"TVT Split (Fixed Test Start: {test_start_date.date()})")
    print(f"{'='*80}")
    print(f"  Train: {len(train_df):4d} ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
    print(f"  Val:   {len(val_df):4d} ({val_df['date'].min().date()} ~ {val_df['date'].max().date()})")
    print(f"  Test:  {len(test_df):4d} ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")
    print(f"{'='*80}\n")
    
    return {'train': train_df, 'val': val_df, 'test': test_df}

def split_walk_forward_method(df, train_start_date, 
                              final_test_start='2025-01-01',
                              initial_train_size=800,    
                              val_size=150,             
                              test_size=150,           
                              step=150,                  
                              gap_size=7):
    """
    Reverse Rolling Walk-Forward Validation
    - 마지막 날짜부터 시작해서 과거로 rolling
    - Train 크기는 고정 (initial_train_size)
    - Final holdout은 2025-01-01부터 고정
    """
    
    df_period = df[df['date'] >= train_start_date].copy()
    df_period = df_period.sort_values('date').reset_index(drop=True)
    
    if isinstance(final_test_start, str):
        final_test_start = pd.to_datetime(final_test_start)
    
    final_test_df = df_period[df_period['date'] >= final_test_start].copy()
    
    total_days = len(df_period)
    min_required_days = initial_train_size + val_size + (gap_size * 2) + test_size
    n_splits = (total_days - min_required_days) // step + 1
    
    print(f"\n{'='*80}")
    print(f"Reverse Rolling Walk-Forward Configuration ")
    print(f"{'='*80}")
    print(f"Total: {len(df_period)} days")
    print(f"Rolling train size: {initial_train_size} days (FIXED)")
    print(f"Val: {val_size} days | Test: {test_size} days")
    print(f"Gap: {gap_size} days | Step: {step} days (BACKWARD)")
    print(f"Target: {n_splits} walk-forward + 1 final holdout")
    print(f"{'='*80}\n")
    
    folds = []
    
    # 역방향 rolling
    for fold_idx in range(n_splits):
        test_end_idx = total_days - (fold_idx * step)
        test_start_idx = test_end_idx - test_size
        
        if test_start_idx < 0:
            break
        
        val_end_idx = test_start_idx - gap_size
        val_start_idx = val_end_idx - val_size
        
        train_end_idx = val_start_idx - gap_size
        train_start_idx = train_end_idx - initial_train_size
        
        if train_start_idx < 0:
            break
        
        train_fold = df_period.iloc[train_start_idx:train_end_idx].copy()
        val_fold = df_period.iloc[val_start_idx:val_end_idx].copy()
        test_fold = df_period.iloc[test_start_idx:test_end_idx].copy()
        
        folds.append({
            'train': train_fold,
            'val': val_fold,
            'test': test_fold,
            'fold_idx': fold_idx + 1,
            'fold_type': 'walk_forward_rolling_reverse'
        })
    
    # 시간순으로 정렬
    folds.reverse()
    for idx, fold in enumerate(folds):
        fold['fold_idx'] = idx + 1
        
        print(f"Fold {fold['fold_idx']} (walk_forward_rolling)")
        print(f"  Train: {len(fold['train']):4d}d  {fold['train']['date'].min().date()} ~ {fold['train']['date'].max().date()}")
        print(f"  Val:   {len(fold['val']):4d}d  {fold['val']['date'].min().date()} ~ {fold['val']['date'].max().date()}")
        print(f"  Test:  {len(fold['test']):4d}d  {fold['test']['date'].min().date()} ~ {fold['test']['date'].max().date()}\n")
    
    # Final holdout
    if len(final_test_df) > 0:
        pre_final_df = df_period[df_period['date'] < final_test_start].copy()
        
        final_val_end_idx = len(pre_final_df)
        final_val_start_idx = final_val_end_idx - val_size
        final_train_end_idx = final_val_start_idx - gap_size
        final_train_start_idx = final_train_end_idx - initial_train_size
        
        if final_train_start_idx < 0:
            final_train_start_idx = 0
        
        final_train_data = pre_final_df.iloc[final_train_start_idx:final_train_end_idx].copy()
        final_val_data = pre_final_df.iloc[final_val_start_idx:final_val_end_idx].copy()
        
        print(f"Fold {len(folds) + 1} (final_holdout)")
        print(f"  Train: {len(final_train_data):4d}d  {final_train_data['date'].min().date()} ~ {final_train_data['date'].max().date()}")
        print(f"  Val:   {len(final_val_data):4d}d  {final_val_data['date'].min().date()} ~ {final_val_data['date'].max().date()}")
        print(f"  Test:  {len(final_test_df):4d}d  {final_test_df['date'].min().date()} ~ {final_test_df['date'].max().date()}\n")
        
        folds.append({
            'train': final_train_data,
            'val': final_val_data,
            'test': final_test_df,
            'fold_idx': len(folds) + 1,
            'fold_type': 'final_holdout'
        })
    
    print(f"{'='*80}")
    print(f"Created {len(folds)} folds total")
    print(f"{'='*80}\n")
    
    return folds


def process_single_split(split_data, target_type='direction', top_n=40, fold_idx=None):
    """
    각 fold를 독립적으로 처리 (feature selection 포함)
    """
    
    train_df = split_data['train']
    val_df = split_data['val']
    test_df = split_data['test']
    fold_type = split_data.get('fold_type', 'unknown')
    
    if fold_idx is not None:
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx} ({fold_type})")
        print(f"{'='*60}")
    
    train_processed, missing_stats = handle_missing_values_paper_based(
        train_df.copy(),
        train_start_date=train_df['date'].min(),
        is_train=True
    )
    
    val_processed = handle_missing_values_paper_based(
        val_df.copy(),
        train_start_date=val_df['date'].min(),
        is_train=False,
        train_stats=missing_stats
    )
    
    test_processed = handle_missing_values_paper_based(
        test_df.copy(),
        train_start_date=test_df['date'].min(),
        is_train=False,
        train_stats=missing_stats
    )
    
    target_cols = ['next_log_return', 'next_direction', 'next_close','next_open']
    
    train_processed = train_processed.dropna(subset=target_cols).reset_index(drop=True)
    val_processed = val_processed.dropna(subset=target_cols).reset_index(drop=True)
    test_processed = test_processed.dropna(subset=target_cols).reset_index(drop=True)

    feature_cols = [col for col in train_processed.columns 
                   if col not in target_cols + ['date']]
    
    X_train = train_processed[feature_cols]
    y_train = train_processed[target_cols]
    
    X_val = val_processed[feature_cols]
    y_val = val_processed[target_cols]
    
    X_test = test_processed[feature_cols]
    y_test = test_processed[target_cols]

    print(f"\n[Feature Selection for Fold {fold_idx}]")
    print(f"Training data shape: {X_train.shape}")
    
    selected_features, selection_stats = select_features_multi_target(
        X_train, 
        y_train, 
        target_type=target_type, 
        top_n=top_n
    )
    
    print(f"Selected {len(selected_features)} features for this fold")
    
    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    X_test_sel = X_test[selected_features]
    
    robust_scaler = RobustScaler()
    standard_scaler = StandardScaler()
    
    X_train_robust = robust_scaler.fit_transform(X_train_sel)
    X_val_robust = robust_scaler.transform(X_val_sel)
    X_test_robust = robust_scaler.transform(X_test_sel)
    
    X_train_standard = standard_scaler.fit_transform(X_train_sel)
    X_val_standard = standard_scaler.transform(X_val_sel)
    X_test_standard = standard_scaler.transform(X_test_sel)
    
    print(f"Scaling completed for Fold {fold_idx}")
    print(f"{'='*60}\n")
    
    result = {
        'train': {
            'X_robust': X_train_robust,
            'X_standard': X_train_standard,
            'X_raw': X_train_sel,
            'y': y_train.reset_index(drop=True), 
            'dates': train_df['date'].reset_index(drop=True) 
        },
        'val': {
            'X_robust': X_val_robust,
            'X_standard': X_val_standard,
            'X_raw': X_val_sel,
            'y': y_val.reset_index(drop=True), 
            'dates': val_df['date'].reset_index(drop=True)  
        },
        'test': {
            'X_robust': X_test_robust,
            'X_standard': X_test_standard,
            'X_raw': X_test_sel,
            'y': y_test.reset_index(drop=True),  
            'dates': test_df['date'].reset_index(drop=True)  
        },
        'scaler': robust_scaler, 
        'stats': {
            'robust_scaler': robust_scaler,
            'standard_scaler': standard_scaler,
            'selected_features': selected_features,
            'selection_stats': selection_stats,
            'target_type': target_type,
            'target_cols': target_cols,
            'fold_type': fold_type,
            'fold_idx': fold_idx
        }
    }
    
    return result



def build_complete_pipeline_corrected(df_raw, train_start_date, 
                                     final_test_start='2025-01-01',
                                     method='tvt', target_type='direction', **kwargs):
    """
    전체 파이프라인 실행 함수
    
    Parameters:
    -----------
    df_raw : DataFrame
        원본 데이터
    train_start_date : str
        학습 데이터 시작 날짜
    final_test_start : str, default='2025-01-01'
        최종 고정 테스트 시작 날짜
        - TVT: 이 날짜부터 마지막까지 테스트
        - Walk-forward: 이 날짜 이전은 walk-forward folds, 이후는 final holdout
    method : str, default='tvt'
        'tvt' 또는 'walk_forward'
    target_type : str, default='direction'
        'direction', 'return', 'price', 'direction_return', 'direction_price'
    **kwargs : dict
        각 method에 필요한 추가 파라미터
    """
    
    df = df_raw.copy()

    df = add_price_lag_features_first(df)
    df = calculate_technical_indicators(df)
    df = add_enhanced_cross_crypto_features(df)
    df = add_volatility_regime_features(df)
    df = add_interaction_features(df)
    df = add_percentile_features(df)
    df = add_normalized_price_lags(df)
    df = create_targets(df)
    df = remove_raw_prices_and_transform(df)
    df = preprocess_non_stationary_features(df)
    df = apply_lag_features(df, news_lag=2, onchain_lag=1)


    pd.set_option('display.max_columns', None)
    df = df.iloc[:-1]  
    
    split_kwargs = {}
    
    if method == 'tvt':
        split_kwargs['test_start_date'] = final_test_start
        if 'train_ratio' in kwargs:
            split_kwargs['train_ratio'] = kwargs['train_ratio']
        if 'val_ratio' in kwargs:
            split_kwargs['val_ratio'] = kwargs['val_ratio']
        splits = split_tvt_method(df, train_start_date, **split_kwargs)
        
    elif method == 'walk_forward':
        split_kwargs['final_test_start'] = final_test_start
        if 'n_splits' in kwargs:
            split_kwargs['n_splits'] = kwargs['n_splits']
        if 'initial_train_size' in kwargs:
            split_kwargs['initial_train_size'] = kwargs['initial_train_size']
        if 'test_size' in kwargs:
            split_kwargs['test_size'] = kwargs['test_size']
        if 'val_size' in kwargs:
            split_kwargs['val_size'] = kwargs['val_size']
        if 'step' in kwargs:
            split_kwargs['step'] = kwargs['step']
        splits = split_walk_forward_method(df, train_start_date, **split_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if method == 'tvt':
        result = process_single_split(
            splits, 
            target_type=target_type,  
            top_n=30,
            fold_idx=1
        )
    else:
        result = [
            process_single_split(
                fold, 
                target_type=target_type,  
                top_n=30,
                fold_idx=fold['fold_idx']
            ) 
            for fold in splits
        ]
    
    return result
