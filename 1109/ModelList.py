class TimeSeriesAugmentation:
    """
    시계열 데이터 증강을 위한 유틸리티 클래스
    """
    
    @staticmethod
    def jittering(X, sigma=0.02):
        """
        가우시안 노이즈 추가
        """
        noise = np.random.normal(0, sigma, X.shape)
        return X + noise
    
    @staticmethod
    def scaling(X, sigma=0.1):
        """
        랜덤 스케일링 적용
        """
        if len(X.shape) == 3:
            factor = np.random.normal(1, sigma, (X.shape[0], 1, X.shape[2]))
        else:
            factor = np.random.normal(1, sigma, (X.shape[0], X.shape[1]))
        return X * factor
    
    @staticmethod
    def magnitude_warping(X, sigma=0.2, num_knots=4):
        """
        진폭 왜곡 적용
        """
        if len(X.shape) == 3:
            seq_len = X.shape[1]
            orig_steps = np.linspace(0, seq_len - 1, num_knots + 2)
            random_warps = np.random.normal(1, sigma, size=(X.shape[0], num_knots + 2, X.shape[2]))
            
            warped_X = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[2]):
                    warper = np.interp(np.arange(seq_len), orig_steps, random_warps[i, :, j])
                    warped_X[i, :, j] = X[i, :, j] * warper
            return warped_X
        else:
            return X * np.random.normal(1, sigma, X.shape)
    
    @staticmethod
    def apply_augmentation(X, method='jittering', **kwargs):
        """
        선택된 증강 기법 적용
        """
        if method == 'jittering':
            return TimeSeriesAugmentation.jittering(X, **kwargs)
        elif method == 'scaling':
            return TimeSeriesAugmentation.scaling(X, **kwargs)
        elif method == 'magnitude_warping':
            return TimeSeriesAugmentation.magnitude_warping(X, **kwargs)
        else:
            return X

class DirectionModels:
    
    @staticmethod
    def random_forest(X_train, y_train, X_val, y_val):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 80, 200),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'min_samples_split': trial.suggest_int('min_samples_split', 40, 70),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 35),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'max_samples': trial.suggest_float('max_samples', 0.6, 0.8),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 40, 100),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.01),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01),
                'random_state': 42,
                'n_jobs': -1,
                'bootstrap': True
            }
            
            model = RandomForestClassifier(**param)
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            
            gap_penalty = max(0, (train_acc - val_acc) - 0.03)
            return val_acc - 1.0 * gap_penalty
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        )
        
        study.optimize(objective, n_trials=30, show_progress_bar=False, n_jobs=1)
        
        best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1, bootstrap=True)
        best_model.fit(X_train, y_train)
        
        train_acc = best_model.score(X_train, y_train)
        val_acc = best_model.score(X_val, y_val)
        print(f"[Random Forest] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")
        
        return best_model
    
    @staticmethod
    def lightgbm(X_train, y_train, X_val, y_val):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 100),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 1.0, log=True),
                'path_smooth': trial.suggest_float('path_smooth', 0.0, 1.0),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.8),
                'bagging_freq': 1,
                'random_state': 42,
                'verbose': -1,
                'force_col_wise': True
            }

            model = LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='binary_logloss',
                callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
            )

            train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            gap_penalty = max(0, (train_acc - val_acc) - 0.03)
            return val_acc - 1.0 * gap_penalty

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=10)
        )

        study.optimize(objective, n_trials=30, show_progress_bar=False)

        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['verbose'] = -1
        best_params['force_col_wise'] = True
        best_params['bagging_freq'] = 1

        final_model = LGBMClassifier(**best_params)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
        )

        train_pred = final_model.predict(X_train)
        val_pred = final_model.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"[LightGBM] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")

        return final_model
    
    @staticmethod
    def xgboost(X_train, y_train, X_val, y_val):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.8),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 20.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 10, 30),
                'gamma': trial.suggest_float('gamma', 0.1, 2.0, log=True),
                'max_delta_step': trial.suggest_float('max_delta_step', 0, 3),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.5),
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
                'eval_metric': 'logloss'
            }

            model = XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            gap_penalty = max(0, (train_acc - val_acc) - 0.03)
            return val_acc - 1.0 * gap_penalty

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=10)
        )

        study.optimize(objective, n_trials=30, show_progress_bar=False)

        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        best_params['tree_method'] = 'hist'
        best_params['eval_metric'] = 'logloss'

        final_model = XGBClassifier(**best_params)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        train_pred = final_model.predict(X_train)
        val_pred = final_model.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"[XGBoost] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")

        return final_model

    @staticmethod
    def histgradient_boosting(X_train, y_train, X_val, y_val):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            params = {
                'max_iter': trial.suggest_int('max_iter', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 25, 70),
                'l2_regularization': trial.suggest_float('l2_regularization', 1.0, 20.0, log=True),
                'max_bins': trial.suggest_int('max_bins', 128, 255),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 40),
                'early_stopping': True,
                'n_iter_no_change': 20,
                'validation_fraction': 0.1,
                'random_state': 42
            }
            
            model = HistGradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            
            gap_penalty = max(0, (train_acc - val_acc) - 0.03)
            return val_acc - 1.0 * gap_penalty
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
        )
        
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        best_model = HistGradientBoostingClassifier(**study.best_params)
        best_model.fit(X_train, y_train)
        
        train_acc = best_model.score(X_train, y_train)
        val_acc = best_model.score(X_val, y_val)
        print(f"[HistGradientBoosting] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")
        
        return best_model
    @staticmethod
    def logistic_regression(X_train, y_train, X_val, y_val):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            param = {
                'C': trial.suggest_float('C', 0.01, 5.0, log=True),
                'penalty': 'l2',
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
                'max_iter': 3000,
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = LogisticRegression(**param)
            model.fit(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            return val_acc
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=6)
        )
        
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        best_model = LogisticRegression(**study.best_params)
        best_model.fit(X_train, y_train)
        
        train_acc = best_model.score(X_train, y_train)
        val_acc = best_model.score(X_val, y_val)
        print(f"[Logistic Regression] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")
        
        return best_model


    @staticmethod
    def adaboost(X_train, y_train, X_val, y_val):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 30, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5),
                'algorithm': 'SAMME',
                'random_state': 42
            }
            
            base_max_depth = trial.suggest_int('base_max_depth', 1, 3)
            base_min_samples_split = trial.suggest_int('base_min_samples_split', 30, 60)
            base_min_samples_leaf = trial.suggest_int('base_min_samples_leaf', 15, 30)
            
            base_estimator = DecisionTreeClassifier(
                max_depth=base_max_depth,
                min_samples_split=base_min_samples_split,
                min_samples_leaf=base_min_samples_leaf,
                max_features='sqrt',
                random_state=42
            )
            
            model = AdaBoostClassifier(estimator=base_estimator, **param)
            model.fit(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            return val_acc
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=6)
        )
        
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        
        best_params = study.best_params
        base_estimator = DecisionTreeClassifier(
            max_depth=best_params['base_max_depth'],
            min_samples_split=best_params['base_min_samples_split'],
            min_samples_leaf=best_params['base_min_samples_leaf'],
            max_features='sqrt',
            random_state=42
        )
        
        best_model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            algorithm='SAMME',
            random_state=42
        )
        best_model.fit(X_train, y_train)
        
        train_acc = best_model.score(X_train, y_train)
        val_acc = best_model.score(X_val, y_val)
        print(f"[AdaBoost] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")
        
        return best_model
    
    @staticmethod
    def catboost(X_train, y_train, X_val, y_val):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'depth': trial.suggest_int('depth', 2, 4),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 8.0, 20.0),
                'subsample': trial.suggest_float('subsample', 0.4, 0.7),
                'rsm': trial.suggest_float('rsm', 0.4, 0.7),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 40, 80),
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 20
            }
            
            model = CatBoostClassifier(**param)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
            
            val_acc = model.score(X_val, y_val)
            return val_acc
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
        )
        
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        
        model = CatBoostClassifier(**study.best_params, random_seed=42, verbose=False)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        print(f"[CatBoost] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")
        
        return model

    @staticmethod
    def gradient_boosting(X_train, y_train, X_val, y_val):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 80, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'subsample': trial.suggest_float('subsample', 0.4, 0.7),
                'min_samples_split': trial.suggest_int('min_samples_split', 40, 80),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 40),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 30, 80),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.02),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.02),
                'validation_fraction': 0.15,
                'n_iter_no_change': 15,
                'tol': 0.001,
                'random_state': 42
            }
            
            model = GradientBoostingClassifier(**param)
            model.fit(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            return val_acc
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=12),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=6)
        )
        
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        
        best_model = GradientBoostingClassifier(**study.best_params)
        best_model.fit(X_train, y_train)
        
        train_acc = best_model.score(X_train, y_train)
        val_acc = best_model.score(X_val, y_val)
        print(f"[Gradient Boosting] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")
        
        return best_model

    @staticmethod
    def stacking_ensemble(X_train, y_train, X_val, y_val):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            xgb_estimators = trial.suggest_int('xgb_estimators', 100, 200)
            xgb_depth = trial.suggest_int('xgb_depth', 3, 5)
            xgb_lr = trial.suggest_float('xgb_lr', 0.01, 0.05, log=True)
            
            lgbm_estimators = trial.suggest_int('lgbm_estimators', 100, 200)
            lgbm_depth = trial.suggest_int('lgbm_depth', 3, 5)
            lgbm_lr = trial.suggest_float('lgbm_lr', 0.01, 0.05, log=True)
            
            meta_C = trial.suggest_float('meta_C', 0.1, 2.0, log=True)
            
            base_learners = [
                ('xgb', XGBClassifier(
                    n_estimators=xgb_estimators,
                    max_depth=xgb_depth,
                    learning_rate=xgb_lr,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    reg_alpha=2.0,
                    reg_lambda=3.0,
                    min_child_weight=10,
                    random_state=42,
                    n_jobs=-1
                )),
                ('lgbm', LGBMClassifier(
                    n_estimators=lgbm_estimators,
                    max_depth=lgbm_depth,
                    learning_rate=lgbm_lr,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    reg_alpha=2.0,
                    reg_lambda=2.0,
                    min_child_samples=60,
                    random_state=42,
                    verbose=-1,
                    force_col_wise=True
                ))
            ]
            
            meta_learner = LogisticRegression(max_iter=3000, C=meta_C, random_state=42, penalty='l2')
            
            model = StackingClassifier(
                estimators=base_learners,
                final_estimator=meta_learner,
                cv=7,
                n_jobs=-1,
                passthrough=False
            )
            
            model.fit(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            return val_acc
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=6),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
        )
        
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        best_params = study.best_params
        base_learners = [
            ('xgb', XGBClassifier(
                n_estimators=best_params['xgb_estimators'],
                max_depth=best_params['xgb_depth'],
                learning_rate=best_params['xgb_lr'],
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=2.0,
                reg_lambda=3.0,
                min_child_weight=10,
                random_state=42,
                n_jobs=-1
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=best_params['lgbm_estimators'],
                max_depth=best_params['lgbm_depth'],
                learning_rate=best_params['lgbm_lr'],
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=2.0,
                reg_lambda=2.0,
                min_child_samples=60,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            ))
        ]
        
        meta_learner = LogisticRegression(max_iter=3000, C=best_params['meta_C'], random_state=42, penalty='l2')
        
        best_model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=7,
            n_jobs=-1,
            passthrough=False
        )
        
        best_model.fit(X_train, y_train)
        
        train_acc = best_model.score(X_train, y_train)
        val_acc = best_model.score(X_val, y_val)
        print(f"[Stacking] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")
        
        return best_model

    @staticmethod
    def lstm(X_train, y_train, X_val, y_val, input_shape):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            units1 = trial.suggest_int('units1', 32, 80, step=16)
            units2 = trial.suggest_int('units2', 16, 48, step=16)
            dropout = trial.suggest_float('dropout', 0.35, 0.55)
            l2_reg = trial.suggest_float('l2_reg', 0.01, 0.15, log=True)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.002, log=True)

            X_aug = TimeSeriesAugmentation.jittering(X_train, sigma=0.015)

            model = Sequential([
                LSTM(units1, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg * 0.5), dropout=dropout, recurrent_dropout=0.0),
                BatchNormalization(),
                LSTM(units2, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg * 0.5), dropout=dropout, recurrent_dropout=0.0),
                BatchNormalization(),
                Dense(16, activation='relu', kernel_regularizer=l2(l2_reg)),
                Dropout(dropout),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, min_delta=1e-4, mode='min')
            history = model.fit(X_aug, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)

            _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            return val_accuracy

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3), pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3))
        study.optimize(objective, n_trials=8, show_progress_bar=False)

        best_params = study.best_params
        X_aug = TimeSeriesAugmentation.jittering(X_train, sigma=0.015)

        model = Sequential([
            LSTM(best_params['units1'], activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(best_params['l2_reg']), recurrent_regularizer=l2(best_params['l2_reg'] * 0.5), dropout=best_params['dropout'], recurrent_dropout=0.0),
            BatchNormalization(),
            LSTM(best_params['units2'], activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(best_params['l2_reg']), recurrent_regularizer=l2(best_params['l2_reg'] * 0.5), dropout=best_params['dropout'], recurrent_dropout=0.0),
            BatchNormalization(),
            Dense(16, activation='relu', kernel_regularizer=l2(best_params['l2_reg'])),
            Dropout(best_params['dropout']),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'], clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-4, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, mode='min', verbose=0)

        model.fit(X_aug, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=0)

        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"[LSTM] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        return model


    @staticmethod
    def bilstm(X_train, y_train, X_val, y_val, input_shape):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            units1 = trial.suggest_int('units1', 24, 64, step=16)
            units2 = trial.suggest_int('units2', 12, 40, step=12)
            dropout = trial.suggest_float('dropout', 0.4, 0.6)
            l2_reg = trial.suggest_float('l2_reg', 0.02, 0.2, log=True)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.002, log=True)

            X_aug = TimeSeriesAugmentation.jittering(X_train, sigma=0.015)

            model = Sequential([
                Bidirectional(LSTM(units1, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg * 0.5), dropout=dropout, recurrent_dropout=0.0), input_shape=input_shape),
                BatchNormalization(),
                Bidirectional(LSTM(units2, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg * 0.5), dropout=dropout, recurrent_dropout=0.0)),
                BatchNormalization(),
                Dense(12, activation='relu', kernel_regularizer=l2(l2_reg)),
                Dropout(dropout),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, min_delta=1e-4, mode='min')
            history = model.fit(X_aug, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)

            _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            return val_accuracy

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3), pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3))
        study.optimize(objective, n_trials=6, show_progress_bar=False)

        best_params = study.best_params
        X_aug = TimeSeriesAugmentation.jittering(X_train, sigma=0.015)

        model = Sequential([
            Bidirectional(LSTM(best_params['units1'], activation='tanh', recurrent_activation='sigmoid', return_sequences=True, kernel_regularizer=l2(best_params['l2_reg']), recurrent_regularizer=l2(best_params['l2_reg'] * 0.5), dropout=best_params['dropout'], recurrent_dropout=0.0), input_shape=input_shape),
            BatchNormalization(),
            Bidirectional(LSTM(best_params['units2'], activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(best_params['l2_reg']), recurrent_regularizer=l2(best_params['l2_reg'] * 0.5), dropout=best_params['dropout'], recurrent_dropout=0.0)),
            BatchNormalization(),
            Dense(12, activation='relu', kernel_regularizer=l2(best_params['l2_reg'])),
            Dropout(best_params['dropout']),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'], clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, min_delta=1e-4, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, mode='min', verbose=0)

        model.fit(X_aug, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=0)

        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"[BiLSTM] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        return model


    @staticmethod
    def gru(X_train, y_train, X_val, y_val, input_shape):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            units1 = trial.suggest_int('units1', 32, 96, step=16)
            units2 = trial.suggest_int('units2', 16, 56, step=16)
            dropout = trial.suggest_float('dropout', 0.35, 0.55)
            l2_reg = trial.suggest_float('l2_reg', 0.01, 0.15, log=True)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.002, log=True)

            X_aug = TimeSeriesAugmentation.jittering(X_train, sigma=0.015)

            model = Sequential([
                GRU(units1, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg * 0.5), dropout=dropout, recurrent_dropout=0.0),
                BatchNormalization(),
                GRU(units2, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg * 0.5), dropout=dropout, recurrent_dropout=0.0),
                BatchNormalization(),
                Dense(16, activation='relu', kernel_regularizer=l2(l2_reg)),
                Dropout(dropout),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, min_delta=1e-4, mode='min')
            history = model.fit(X_aug, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)

            _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            return val_accuracy

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3), pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3))
        study.optimize(objective, n_trials=8, show_progress_bar=False)

        best_params = study.best_params
        X_aug = TimeSeriesAugmentation.jittering(X_train, sigma=0.015)

        model = Sequential([
            GRU(best_params['units1'], activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(best_params['l2_reg']), recurrent_regularizer=l2(best_params['l2_reg'] * 0.5), dropout=best_params['dropout'], recurrent_dropout=0.0),
            BatchNormalization(),
            GRU(best_params['units2'], activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(best_params['l2_reg']), recurrent_regularizer=l2(best_params['l2_reg'] * 0.5), dropout=best_params['dropout'], recurrent_dropout=0.0),
            BatchNormalization(),
            Dense(16, activation='relu', kernel_regularizer=l2(best_params['l2_reg'])),
            Dropout(best_params['dropout']),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'], clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-4, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, mode='min', verbose=0)

        model.fit(X_aug, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=0)

        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"[GRU] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        return model
