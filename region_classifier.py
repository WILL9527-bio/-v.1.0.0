import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import logging
import datetime
import os
warnings.filterwarnings('ignore')

class ModelLogger:
    """模型训练和预测的日志记录器"""

    def __init__(self, log_file="model_training.log"):
        self.log_file = log_file
        self.setup_logger()

    def setup_logger(self):
        """设置日志配置"""
        # 创建logs目录
        if not os.path.exists('logs'):
            os.makedirs('logs')

        log_path = os.path.join('logs', self.log_file)

        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_data_info(self, X, y, data_type="训练"):
        """记录数据信息"""
        unique_labels, counts = np.unique(y, return_counts=True)
        self.logger.info(f"=== {data_type}数据信息 ===")
        self.logger.info(f"样本数: {len(X)}, 特征数: {X.shape[1]}")
        self.logger.info(f"类别数: {len(unique_labels)}")
        self.logger.info(f"类别分布: {dict(zip(unique_labels, counts))}")

        # 数据质量检查
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            self.logger.warning(f"发现 {nan_count} 个NaN值")

        # 特征统计
        self.logger.info(f"特征值范围: {X.min():.6f} ~ {X.max():.6f}")
        self.logger.info(f"特征均值: {X.mean():.6f}, 标准差: {X.std():.6f}")

    def log_training_start(self, optimize_hyperparams):
        """记录训练开始"""
        self.logger.info("=" * 50)
        self.logger.info("开始模型训练")
        self.logger.info(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"超参数优化: {'是' if optimize_hyperparams else '否'}")

    def log_preprocessing(self, original_shape, processed_shape):
        """记录预处理信息"""
        self.logger.info(f"预处理: {original_shape} -> {processed_shape}")
        reduction_ratio = (1 - processed_shape[1] / original_shape[1]) * 100
        self.logger.info(f"维度降低: {reduction_ratio:.1f}%")

    def log_hyperparameter_optimization(self, param_type, best_params, best_score):
        """记录超参数优化结果"""
        self.logger.info(f"{param_type}优化完成:")
        self.logger.info(f"  最佳参数: {best_params}")
        self.logger.info(f"  最佳分数: {best_score:.4f}")

    def log_training_complete(self, training_time):
        """记录训练完成"""
        self.logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")

    def log_prediction_start(self, n_samples):
        """记录预测开始"""
        self.logger.info(f"开始预测 {n_samples} 个样本")

    def log_prediction_results(self, predictions, probabilities):
        """记录预测结果"""
        max_probs = probabilities.max(axis=1)
        self.logger.info(f"预测完成:")
        self.logger.info(f"  平均置信度: {max_probs.mean():.3f}")
        self.logger.info(f"  最高置信度: {max_probs.max():.3f}")
        self.logger.info(f"  最低置信度: {max_probs.min():.3f}")

        # 预测分布
        unique_preds, counts = np.unique(predictions, return_counts=True)
        pred_dist = dict(zip(unique_preds, counts))
        self.logger.info(f"  预测分布: {pred_dist}")

    def log_evaluation_results(self, results):
        """记录评估结果"""
        self.logger.info("=== 模型评估结果 ===")
        for metric, value in results.items():
            if isinstance(value, dict):
                self.logger.info(f"{metric}:")
                for k, v in value.items():
                    self.logger.info(f"  {k}: {v:.4f}")
            else:
                self.logger.info(f"{metric}: {value:.4f}")

    def log_error(self, error_msg, exception=None):
        """记录错误信息"""
        self.logger.error(f"错误: {error_msg}")
        if exception:
            self.logger.error(f"异常详情: {str(exception)}")

    def log_warning(self, warning_msg):
        """记录警告信息"""
        self.logger.warning(warning_msg)

class SmallSampleRegionClassifier:
    def __init__(self, enable_logging=True):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10, whiten=True)
        self.label_encoder = LabelEncoder()
        self.models = self._init_models()
        self.ensemble = None
        self.use_simple_knn = True

        # 初始化日志记录器
        if enable_logging:
            self.logger = ModelLogger()
        else:
            self.logger = None
        self.feature_importance = None
        self.component_names = None

    def _init_models(self):
        return {
            'knn1': KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
            'knn_cos': KNeighborsClassifier(n_neighbors=1, metric='cosine'),
            'knn3': KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        }

    def _extract_advanced_features(self, X):
        # 基础统计特征
        stats = np.column_stack([
            np.mean(X, axis=1),
            np.std(X, axis=1),
            np.max(X, axis=1),
            np.min(X, axis=1),
            np.median(X, axis=1)
        ])

        # 比例特征 (相对含量)
        X_sum = np.sum(X, axis=1, keepdims=True)
        X_ratio = X / (X_sum + 1e-8)


        ratios = []
        for i in range(min(10, X.shape[1])):
            for j in range(i+1, min(10, X.shape[1])):
                ratio = X[:, i] / (X[:, j] + 1e-8)
                ratios.append(ratio)
        ratio_features = np.column_stack(ratios) if ratios else np.empty((X.shape[0], 0))

        # 分布特征
        skewness = []
        kurtosis = []
        for i in range(X.shape[0]):
            row = X[i, :]
            skew = np.mean(((row - np.mean(row)) / (np.std(row) + 1e-8)) ** 3)
            kurt = np.mean(((row - np.mean(row)) / (np.std(row) + 1e-8)) ** 4) - 3
            skewness.append(skew)
            kurtosis.append(kurt)

        distribution_features = np.column_stack([skewness, kurtosis])

        return np.hstack([X, X_ratio, stats, ratio_features, distribution_features])

    def preprocess_data(self, X, y=None, fit=True):
        X_clean = np.nan_to_num(X, nan=0.0)

        if fit:
            X_features = self._extract_advanced_features(X_clean)
            X_scaled = self.scaler.fit_transform(X_features)
            X_pca = self.pca.fit_transform(X_scaled)
            if y is not None:
                y_encoded = self.label_encoder.fit_transform(y)
                return X_pca, y_encoded
            return X_pca
        else:
            X_features = self._extract_advanced_features(X_clean)
            X_scaled = self.scaler.transform(X_features)
            X_pca = self.pca.transform(X_scaled)
            return X_pca

    def train(self, X, y, optimize_hyperparams=True):
        """
        训练模型，可选择是否优化超参数

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        optimize_hyperparams : bool, default=True
            是否优化超参数
        """
        import time
        start_time = time.time()

        # 记录训练开始
        if self.logger:
            self.logger.log_training_start(optimize_hyperparams)
            self.logger.log_data_info(X, y, "训练")

        n_samples = len(X)
        n_unique_labels = len(np.unique(y))

        # 样本数检查和警告
        print(f"训练数据信息：样本数={n_samples}, 类别数={n_unique_labels}")

        if n_samples < 10:
            warning_msg = "样本数过少(< 10)，模型性能可能不稳定"
            print(f"⚠️  警告：{warning_msg}")
            if self.logger:
                self.logger.log_warning(warning_msg)

        if n_samples == n_unique_labels:
            warning_msg = "每个类别只有1个样本，这是极小样本学习问题"
            print(f"⚠️  警告：{warning_msg}")
            print("   建议：考虑收集更多数据或使用其他机器学习方法")
            if self.logger:
                self.logger.log_warning(warning_msg)

        if n_samples < 50:
            info_msg = "样本数较少，将使用留一法交叉验证(LOOCV)进行评估"
            print(f"ℹ️  提示：{info_msg}")
            if self.logger:
                self.logger.logger.info(info_msg)

        # 如果选择优化超参数，先进行参数优化
        if optimize_hyperparams:
            print("\n=== 开始超参数优化 ===")

            # 优化PCA组件数
            print("1. 优化PCA组件数...")
            pca_results = self.optimize_pca_components(X, y)
            print(f"   最优PCA组件数: {pca_results['best_n_components']}")
            if self.logger:
                self.logger.log_hyperparameter_optimization(
                    "PCA",
                    {"n_components": pca_results['best_n_components']},
                    pca_results['best_score']
                )

            # 优化K值
            print("2. 优化K值...")
            k_results = self.optimize_k_values(X, y)
            print("   最优K值:")
            for metric, result in k_results.items():
                print(f"     {metric}: K={result['best_k']}, 分数={result['best_score']:.3f}")
                if self.logger:
                    self.logger.log_hyperparameter_optimization(
                        f"K值({metric})",
                        {"k": result['best_k']},
                        result['best_score']
                    )
        else:
            print("\n=== 跳过超参数优化，使用默认参数 ===")

        # 使用优化后的参数进行数据预处理
        original_shape = X.shape
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        print(f"预处理后特征维度: {X_processed.shape}")

        if self.logger:
            self.logger.log_preprocessing(original_shape, X_processed.shape)

        # 创建优化后的模型
        if hasattr(self, 'optimized_models') and self.optimized_models:
            # 如果已经有优化的模型，使用它们
            print("使用优化后的模型参数")
            self.knn_euclidean = self.optimized_models['knn_euclidean']
            self.knn_manhattan = self.optimized_models['knn_manhattan']
            self.knn_cosine = self.optimized_models['knn_cosine']
        else:
            # 否则使用默认值
            print("使用默认模型参数")
            self.knn_euclidean = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
            self.knn_manhattan = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
            self.knn_cosine = KNeighborsClassifier(n_neighbors=1, metric='cosine')

        # 训练模型
        print("训练模型...")
        self.knn_euclidean.fit(X_processed, y_processed)
        self.knn_manhattan.fit(X_processed, y_processed)
        self.knn_cosine.fit(X_processed, y_processed)

        self.X_train = X_processed
        self.y_train = y_processed
        self.y_train_original = y  # 保存原始标签用于预测时的类别查找

        # 记录训练完成
        training_time = time.time() - start_time
        print("✅ 模型训练完成")

        if self.logger:
            self.logger.log_training_complete(training_time)

        return self

    def _create_default_k_results(self):
        """创建默认的K值优化结果"""
        return {
            'euclidean': {'best_k': 1, 'best_score': 0.0, 'all_results': {1: {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}}},
            'manhattan': {'best_k': 1, 'best_score': 0.0, 'all_results': {1: {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}}},
            'cosine': {'best_k': 1, 'best_score': 0.0, 'all_results': {1: {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}}}
        }

    def optimize_k_values(self, X, y, k_range=None, cv_folds=3, scoring='f1_macro'):
        """
        使用交叉验证优化K值

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        k_range : list, optional
            K值搜索范围，默认为1到min(5, n_samples-1)
        cv_folds : int, default=3
            交叉验证折数
        scoring : str, default='f1_macro'
            评估指标

        Returns:
        --------
        dict : 包含优化结果的字典
        """
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        n_samples = len(X_processed)

        if k_range is None:
            # 对于16地区×3样本的数据集，每类只有3个样本，K值应该限制为1-2
            n_unique_labels = len(np.unique(y))
            samples_per_class = n_samples // n_unique_labels
            if samples_per_class <= 3:
                # 每类样本数较少时，K值限制为1-2
                max_k = min(2, n_samples - 1)
            else:
                # 原有逻辑
                max_k = min(5, n_samples - 1)
            k_range = list(range(1, max_k + 1))

        # 过滤掉超过样本数的K值
        k_range = [k for k in k_range if k < n_samples]

        if not k_range:
            print(f"警告：样本数({n_samples})过少，无法进行K值优化，使用默认K=1")
            return self._create_default_k_results()

        # 检查每个类别的样本数
        n_unique_labels = len(np.unique(y))
        samples_per_class = n_samples // n_unique_labels

        # 对于极小样本或每类样本数很少，强制使用LOOCV
        if n_samples <= 20 or samples_per_class < cv_folds:
            cv = LeaveOneOut()
            print(f"样本数({n_samples})较少或每类样本数({samples_per_class})不足，使用留一法交叉验证(LOOCV)")
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        optimization_results = {}

        # 测试不同距离度量的KNN
        metrics = ['euclidean', 'manhattan', 'cosine']

        for metric in metrics:
            metric_results = {}
            best_k = 1
            best_score = 0

            for k in k_range:
                if k >= len(X_processed):
                    continue

                knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                scores = cross_val_score(knn, X_processed, y_processed, cv=cv, scoring=scoring)

                mean_score = scores.mean()
                std_score = scores.std()

                metric_results[k] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'scores': scores
                }

                if mean_score > best_score:
                    best_score = mean_score
                    best_k = k

            optimization_results[metric] = {
                'best_k': best_k,
                'best_score': best_score,
                'all_results': metric_results
            }

        # 更新模型的K值
        self.optimized_models = {
            'knn_euclidean': KNeighborsClassifier(n_neighbors=optimization_results['euclidean']['best_k'], metric='euclidean'),
            'knn_manhattan': KNeighborsClassifier(n_neighbors=optimization_results['manhattan']['best_k'], metric='manhattan'),
            'knn_cosine': KNeighborsClassifier(n_neighbors=optimization_results['cosine']['best_k'], metric='cosine')
        }

        return optimization_results

    def evaluate_with_loocv(self, X, y):
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)

        loo = LeaveOneOut()
        results = {}

        for name, model in self.models.items():
            scores = cross_val_score(model, X_processed, y_processed,
                                   cv=loo, scoring='f1_macro')
            results[name] = {
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'scores': scores
            }

        ensemble_scores = cross_val_score(self.ensemble, X_processed, y_processed,
                                        cv=loo, scoring='f1_macro')
        results['ensemble'] = {
            'mean_f1': ensemble_scores.mean(),
            'std_f1': ensemble_scores.std(),
            'scores': ensemble_scores
        }

        return results

    def evaluate_with_cv(self, X, y, cv_folds=3, scoring_metrics=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']):
        """
        使用k-fold交叉验证评估模型

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        cv_folds : int, default=3
            交叉验证折数
        scoring_metrics : list, default=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']
            评估指标列表

        Returns:
        --------
        dict : 包含评估结果的字典
        """
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        n_samples = len(X_processed)

        # 检查每个类别的样本数
        n_unique_labels = len(np.unique(y))
        samples_per_class = n_samples // n_unique_labels

        # 自动选择合适的交叉验证策略
        if n_samples <= 20 or samples_per_class < cv_folds:
            cv = LeaveOneOut()
            print(f"模型评估：样本数({n_samples})较少或每类样本数({samples_per_class})不足，使用留一法交叉验证")
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            print(f"模型评估：使用{cv_folds}折分层交叉验证")

        results = {}

        for name, model in self.models.items():
            model_results = {}

            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(model, X_processed, y_processed, cv=cv, scoring=metric)
                    model_results[metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores
                    }
                except Exception as e:
                    print(f"评估指标{metric}在模型{name}上失败: {e}")
                    model_results[metric] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'scores': []
                    }

            results[name] = model_results

        return results

    def optimize_pca_components(self, X, y, n_components_range=None, cv_folds=3, scoring='f1_macro'):
        """
        优化PCA组件数

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        n_components_range : list, optional
            PCA组件数搜索范围
        cv_folds : int, default=3
            交叉验证折数
        scoring : str, default='f1_macro'
            评估指标

        Returns:
        --------
        dict : 包含优化结果的字典
        """
        X_clean = np.nan_to_num(X, nan=0.0)
        X_features = self._extract_advanced_features(X_clean)
        X_scaled = self.scaler.fit_transform(X_features)

        n_samples, n_features = X_scaled.shape

        if n_components_range is None:
            # 对于16地区×3样本的数据集（48样本），PCA组件数范围调整为2-15
            n_unique_labels = len(np.unique(y))
            if n_unique_labels >= 10:  # 对于多类别问题（如16类）
                max_components = min(min(n_samples, n_features) - 1, 15)
                n_components_range = list(range(2, max_components + 1))
            else:
                # 原有逻辑：对于类别较少的问题
                max_components = min(min(n_samples, n_features) - 1, 10)
                n_components_range = list(range(2, max_components + 1))

        # 过滤掉超过样本数或特征数的组件数
        n_components_range = [n for n in n_components_range if n < min(n_samples, n_features)]

        if not n_components_range:
            print(f"警告：样本数({n_samples})或特征数({n_features})过少，无法进行PCA优化，使用默认组件数")
            return {'best_n_components': min(5, min(n_samples, n_features) - 1), 'best_score': 0.0, 'all_results': {}}

        # 对于极小样本或每类样本数较少的情况，强制使用LOOCV
        n_unique_labels = len(np.unique(y))
        samples_per_class = n_samples // n_unique_labels

        if n_samples <= 20 or samples_per_class <= 5:
            cv = LeaveOneOut()
            print(f"PCA优化：样本数({n_samples})较少或每类样本数({samples_per_class})较少，使用留一法交叉验证(LOOCV)")
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            print(f"PCA优化：使用{cv_folds}折分层交叉验证")

        y_encoded = self.label_encoder.fit_transform(y)

        pca_results = {}
        best_n_components = min(n_components_range) if n_components_range else 2
        best_score = 0

        print(f"PCA优化：测试组件数范围 {n_components_range}")

        for n_comp in n_components_range:
            try:
                # 创建临时PCA
                temp_pca = PCA(n_components=n_comp, whiten=True)
                X_pca = temp_pca.fit_transform(X_scaled)

                # 测试不同的KNN模型
                models_to_test = {
                    'euclidean': KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
                    'manhattan': KNeighborsClassifier(n_neighbors=1, metric='manhattan'),
                    'cosine': KNeighborsClassifier(n_neighbors=1, metric='cosine')
                }

                component_scores = {}
                total_score = 0

                for metric_name, model in models_to_test.items():
                    scores = cross_val_score(model, X_pca, y_encoded, cv=cv, scoring=scoring)
                    component_scores[metric_name] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores
                    }
                    total_score += scores.mean()

                avg_score = total_score / len(models_to_test)

                pca_results[n_comp] = {
                    'avg_score': avg_score,
                    'individual_scores': component_scores,
                    'variance_explained': temp_pca.explained_variance_ratio_.sum()
                }

                if avg_score > best_score:
                    best_score = avg_score
                    best_n_components = n_comp

            except Exception as e:
                print(f"PCA组件数{n_comp}测试失败: {e}")
                continue

        # 更新PCA组件数
        if best_n_components > 0:
            self.pca = PCA(n_components=best_n_components, whiten=True)
            print(f"PCA优化完成：最优组件数 = {best_n_components}")

        return {
            'best_n_components': best_n_components,
            'best_score': best_score,
            'all_results': pca_results
        }

    def _calculate_adaptive_weights(self, distances_e, distances_m, distances_c):
        # 基于距离的自适应权重
        inv_dist_e = 1.0 / (distances_e + 1e-8)
        inv_dist_m = 1.0 / (distances_m + 1e-8)
        inv_dist_c = 1.0 / (distances_c + 1e-8)

        total_inv = inv_dist_e + inv_dist_m + inv_dist_c
        weight_e = inv_dist_e / total_inv
        weight_m = inv_dist_m / total_inv
        weight_c = inv_dist_c / total_inv

        return weight_e, weight_m, weight_c

    def predict(self, X):
        # 记录预测开始
        if self.logger:
            self.logger.log_prediction_start(len(X))

        X_processed = self.preprocess_data(X, fit=False)

        pred_euclidean = self.knn_euclidean.predict(X_processed)
        pred_manhattan = self.knn_manhattan.predict(X_processed)
        pred_cosine = self.knn_cosine.predict(X_processed)

        distances_euclidean, _ = self.knn_euclidean.kneighbors(X_processed)
        distances_manhattan, _ = self.knn_manhattan.kneighbors(X_processed)
        distances_cosine, _ = self.knn_cosine.kneighbors(X_processed)

        final_predictions = []
        for i in range(len(X_processed)):
            dist_e = distances_euclidean[i][0]
            dist_m = distances_manhattan[i][0]
            dist_c = distances_cosine[i][0]

            weight_e, weight_m, weight_c = self._calculate_adaptive_weights(dist_e, dist_m, dist_c)

            # 如果某个距离明显最小，使用该预测
            if weight_e > 0.6:
                final_predictions.append(pred_euclidean[i])
            elif weight_m > 0.6:
                final_predictions.append(pred_manhattan[i])
            elif weight_c > 0.6:
                final_predictions.append(pred_cosine[i])
            else:
                # 否则使用加权投票
                votes = [pred_euclidean[i], pred_manhattan[i], pred_cosine[i]]
                if len(set(votes)) == 1:
                    final_predictions.append(votes[0])
                else:
                    final_predictions.append(pred_euclidean[i])  # 默认欧氏距离

        predictions = self.label_encoder.inverse_transform(final_predictions)

        # 记录预测结果
        if self.logger:
            probabilities = self.predict_proba(X)
            self.logger.log_prediction_results(predictions, probabilities)

        return predictions

    def predict_proba(self, X):
        X_processed = self.preprocess_data(X, fit=False)

        distances_euclidean, indices_euclidean = self.knn_euclidean.kneighbors(X_processed)
        distances_manhattan, indices_manhattan = self.knn_manhattan.kneighbors(X_processed)
        distances_cosine, indices_cosine = self.knn_cosine.kneighbors(X_processed)

        proba = np.zeros((len(X_processed), len(self.label_encoder.classes_)))

        for i in range(len(X_processed)):
            dist_e = distances_euclidean[i][0]
            dist_m = distances_manhattan[i][0]
            dist_c = distances_cosine[i][0]

            conf_e = max(0.5, 1.0 / (1.0 + dist_e * 2))
            conf_m = max(0.5, 1.0 / (1.0 + dist_m * 1.5))
            conf_c = max(0.5, 1.0 / (1.0 + dist_c * 3))

            avg_confidence = (conf_e + conf_m + conf_c) / 3
            final_confidence = min(0.95, max(0.6, avg_confidence))

            # 获取最近邻的样本索引
            sample_idx_e = indices_euclidean[i][0]

            # 边界检查，确保样本索引不越界
            if sample_idx_e >= len(self.y_train):
                print(f"警告：样本索引{sample_idx_e}超出训练数据范围，训练样本数为{len(self.y_train)}")
                sample_idx_e = 0  # 使用第一个样本作为默认值

            # 获取该样本对应的类别索引
            # 使用原始标签而不是编码后的标签
            predicted_class = self.y_train_original[sample_idx_e]
            class_matches = np.where(self.label_encoder.classes_ == predicted_class)[0]

            if len(class_matches) == 0:
                print(f"警告：未找到类别'{predicted_class}'，可用类别: {self.label_encoder.classes_}")
                class_idx = 0  # 使用第一个类别作为默认值
            else:
                class_idx = class_matches[0]

            proba[i, class_idx] = final_confidence
            remaining = (1.0 - final_confidence) / (len(self.label_encoder.classes_) - 1)
            proba[i, :] += remaining
            proba[i, class_idx] = final_confidence

        return proba

    def get_feature_importance(self):
        if self.pca is None:
            return None

        # PCA成分的重要性
        pca_importance = np.abs(self.pca.components_).mean(axis=0)

        # 如果有成分名称，返回带名称的重要性
        if self.component_names is not None:
            importance_dict = {}
            for i, name in enumerate(self.component_names[:len(pca_importance)]):
                importance_dict[name] = pca_importance[i]
            return importance_dict

        return pca_importance

    def explain_prediction(self, X, sample_idx=0):
        X_processed = self.preprocess_data(X, fit=False)

        distances_e, indices_e = self.knn_euclidean.kneighbors([X_processed[sample_idx]])
        distances_m, indices_m = self.knn_manhattan.kneighbors([X_processed[sample_idx]])
        distances_c, indices_c = self.knn_cosine.kneighbors([X_processed[sample_idx]])

        explanation = {
            'euclidean_distance': distances_e[0][0],
            'manhattan_distance': distances_m[0][0],
            'cosine_distance': distances_c[0][0],
            'nearest_region_euclidean': self.label_encoder.inverse_transform([self.y_train[indices_e[0][0]]])[0],
            'nearest_region_manhattan': self.label_encoder.inverse_transform([self.y_train[indices_m[0][0]]])[0],
            'nearest_region_cosine': self.label_encoder.inverse_transform([self.y_train[indices_c[0][0]]])[0],
        }

        weight_e, weight_m, weight_c = self._calculate_adaptive_weights(
            distances_e[0][0], distances_m[0][0], distances_c[0][0]
        )
        explanation['distance_weights'] = {
            'euclidean': weight_e,
            'manhattan': weight_m,
            'cosine': weight_c
        }

        return explanation

    def generate_optimization_report(self, X, y, save_to_file=False, filename='optimization_report.txt'):
        """
        生成超参数优化报告

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练特征
        y : array-like, shape = [n_samples]
            训练标签
        save_to_file : bool, default=False
            是否保存到文件
        filename : str, default='optimization_report.txt'
            报告文件名

        Returns:
        --------
        str : 优化报告内容
        """
        n_samples = len(X)
        n_unique_labels = len(np.unique(y))

        report = []
        report.append("=" * 60)
        report.append("药材地区分类器 - 超参数优化报告")
        report.append("=" * 60)

        # 数据集信息
        report.append(f"\n📊 数据集信息:")
        report.append(f"   样本数: {n_samples}")
        report.append(f"   类别数: {n_unique_labels}")
        report.append(f"   特征数: {X.shape[1]}")

        # 样本量分析
        if n_samples == n_unique_labels:
            report.append(f"\n⚠️  极小样本警告:")
            report.append(f"   每个类别只有1个样本，这是极小样本学习问题")
            report.append(f"   建议：考虑收集更多数据或使用其他机器学习方法")
        elif n_samples < 50:
            report.append(f"\n⚠️  小样本提示:")
            report.append(f"   样本数较少，使用留一法交叉验证(LOOCV)进行评估")

        # 交叉验证策略
        report.append(f"\n🔍 交叉验证策略:")
        if n_samples <= 20:
            report.append(f"   使用留一法交叉验证(LOOCV) - 适合极小样本")
        else:
            report.append(f"   使用5折分层交叉验证")

        # PCA组件数优化
        report.append("\n1. PCA组件数优化结果:")
        report.append("-" * 40)

        try:
            pca_results = self.optimize_pca_components(X, y)
            report.append(f"   最优PCA组件数: {pca_results['best_n_components']}")
            report.append(f"   最优分数: {pca_results['best_score']:.4f}")

            if pca_results['all_results']:
                report.append("\n   详细结果:")
                for n_comp, result in list(pca_results['all_results'].items())[:5]:  # 只显示前5个
                    report.append(f"     组件数{n_comp}: 平均分数={result['avg_score']:.4f}, 方差解释率={result['variance_explained']:.4f}")
        except Exception as e:
            report.append(f"   PCA优化失败: {e}")

        # K值优化
        report.append("\n2. K值优化结果:")
        report.append("-" * 40)

        try:
            k_results = self.optimize_k_values(X, y)
            for metric, result in k_results.items():
                report.append(f"\n   {metric}距离:")
                report.append(f"     最优K值: {result['best_k']}")
                report.append(f"     最优分数: {result['best_score']:.4f}")

                if result['all_results']:
                    report.append("     详细结果:")
                    for k, k_result in list(result['all_results'].items())[:3]:  # 只显示前3个
                        report.append(f"       K={k}: {k_result['mean_score']:.4f}±{k_result['std_score']:.4f}")
        except Exception as e:
            report.append(f"   K值优化失败: {e}")

        # 交叉验证结果
        report.append("\n3. 交叉验证评估结果:")
        report.append("-" * 40)

        try:
            cv_results = self.evaluate_with_cv(X, y)
            for model_name, metrics in cv_results.items():
                report.append(f"\n   {model_name}:")
                for metric_name, metric_result in metrics.items():
                    if metric_result['mean'] > 0:  # 只显示成功的指标
                        report.append(f"     {metric_name}: {metric_result['mean']:.4f}±{metric_result['std']:.4f}")
        except Exception as e:
            report.append(f"   交叉验证评估失败: {e}")

        # 建议和总结
        report.append("\n4. 优化建议和总结:")
        report.append("-" * 40)

        try:
            # 分析结果并给出建议
            if 'k_results' in locals():
                best_model = max(k_results.items(), key=lambda x: x[1]['best_score'])
                report.append(f"   🎯 推荐配置:")
                report.append(f"     距离度量: {best_model[0]}")
                report.append(f"     K值: {best_model[1]['best_k']}")

                if 'pca_results' in locals():
                    report.append(f"     PCA组件数: {pca_results['best_n_components']}")

            # 极小样本特殊建议
            if n_samples <= 20:
                report.append(f"\n   💡 极小样本建议:")
                report.append(f"     - 当前样本数({n_samples})较少，模型泛化能力有限")
                report.append(f"     - 建议收集更多训练数据提高模型稳定性")
                report.append(f"     - 可考虑使用数据增强技术扩充样本")
                report.append(f"     - 或尝试迁移学习、元学习等方法")

        except Exception as e:
            report.append(f"   建议生成失败: {e}")

        report.append(f"\n" + "=" * 60)
        report.append(f"报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report_text = "\n".join(report)

        if save_to_file:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 优化报告已保存到: {filename}")

        return report_text

def load_data(train_file_path, test_file_path=None):
    """
    加载训练数据和测试数据

    Parameters:
    -----------
    train_file_path : str
        训练数据文件路径
    test_file_path : str, optional
        测试数据文件路径，如果为None则只加载训练数据

    Returns:
    --------
    tuple : (X_train, y_train, X_test, test_regions, test_samples) 或 (X_train, y_train)
    """
    # 加载训练数据
    train_data = pd.read_excel(train_file_path, sheet_name=0, header=None)

    # 提取特征数据（从第3行开始，第2列到第41列，共40个特征）
    X_train = train_data.iloc[2:, 1:41].values.astype(float)

    # 提取地区标签（第1列），并提取地区代码
    raw_labels = train_data.iloc[2:, 0].values
    y_train = []
    for label in raw_labels:
        if isinstance(label, str) and '-' in label:
            # 提取地区代码 (如 Y-FJ-1 -> Y-FJ)
            parts = label.split('-')
            if len(parts) >= 3:
                region_code = '-'.join(parts[:2])  # 取前两部分作为地区代码
            else:
                region_code = label
        else:
            region_code = 'Unknown'
        y_train.append(region_code)

    y_train = np.array(y_train)

    print(f"训练数据加载完成:")
    print(f"  样本数: {X_train.shape[0]}")
    print(f"  特征数: {X_train.shape[1]}")
    print(f"  地区数: {len(np.unique(y_train))}")
    print(f"  地区列表: {list(np.unique(y_train))}")

    # 如果没有提供测试数据文件，只返回训练数据
    if test_file_path is None:
        return X_train, y_train

    # 加载测试数据
    try:
        test_data = pd.read_excel(test_file_path, sheet_name=0, header=None)

        # 测试集样本ID（第1列，从第3行开始）
        test_sample_ids = test_data.iloc[2:, 0].dropna().tolist()

        # 测试集地区标签（第2列，从第3行开始）
        test_region_labels = test_data.iloc[2:, 1].dropna().tolist()

        # 从地区标签中提取地区代码
        test_regions = []
        for region_label in test_region_labels:
            if isinstance(region_label, str) and '-' in region_label:
                # 提取地区代码 (如 "Y-FJ aver" -> "Y-FJ")
                parts = region_label.strip().replace(' aver', '').split('-')
                if len(parts) >= 2:
                    region_code = '-'.join(parts[:2])
                else:
                    region_code = parts[0]
            else:
                region_code = 'Unknown'
            test_regions.append(region_code)

        # 测试集特征数据（第3列开始，从第3行开始，取40个特征以匹配训练集）
        X_test = test_data.iloc[2:, 2:42].values.astype(float)

        print(f"测试数据加载完成:")
        print(f"  测试样本数: {X_test.shape[0]}")
        print(f"  特征数: {X_test.shape[1]}")
        print(f"  地区数: {len(set(test_regions))}")
        print(f"  地区列表: {list(set(test_regions))}")
        print(f"  样本ID: {test_sample_ids}")

        return X_train, y_train, X_test, test_regions, test_sample_ids

    except Exception as e:
        print(f"测试数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return X_train, y_train, np.empty((0, 40)), [], []

def load_data_legacy(file_path):
    """
    兼容旧版本的数据加载函数（用于旧格式数据）
    """
    train_data = pd.read_excel(file_path, sheet_name=0)
    test_data = pd.read_excel(file_path, sheet_name=1)

    X_train = train_data.iloc[1:, 2:40].values.astype(float)
    y_train = train_data.iloc[1:, 1].values

    test_samples = []
    test_regions = []

    for col_idx in range(2, test_data.shape[1]):
        col_name = test_data.columns[col_idx]
        if 'aver' not in col_name:
            try:
                sample = pd.to_numeric(test_data.iloc[:38, col_idx], errors='coerce').values
                if not np.isnan(sample).all() and len(sample) == 38:
                    test_samples.append(sample)
                    region_parts = col_name.split('-')
                    if len(region_parts) >= 2:
                        region_name = f"{region_parts[0]}-{region_parts[1]}"
                    else:
                        region_name = col_name
                    test_regions.append(region_name)
            except:
                continue

    X_test = np.array(test_samples)

    return X_train, y_train, X_test, test_regions

def bootstrap_evaluation(classifier, X, y, n_bootstrap=100):
    n_samples = len(X)
    f1_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        classifier_boot = SmallSampleRegionClassifier()
        classifier_boot.train(X_boot, y_boot)

        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        if len(oob_indices) > 0:
            X_oob = X[oob_indices]
            y_oob = y[oob_indices]
            y_pred = classifier_boot.predict(X_oob)
            f1 = f1_score(y_oob, y_pred, average='macro')
            f1_scores.append(f1)

    return np.array(f1_scores)

def main():
    # 使用新的数据加载函数
    X_train, y_train = load_data('训练集.xlsx')

    print(f"训练数据: {X_train.shape}, 地区数: {len(np.unique(y_train))}")

    classifier = SmallSampleRegionClassifier()

    # 训练模型（包含超参数优化）
    print("\n=== 开始训练和优化 ===")
    classifier.train(X_train, y_train, optimize_hyperparams=True)

    # 生成优化报告
    print("\n=== 生成优化报告 ===")
    report = classifier.generate_optimization_report(X_train, y_train, save_to_file=True)
    print("\n" + report)

    # 使用新的交叉验证方法评估
    print("\n=== 标准交叉验证评估结果 ===")
    cv_results = classifier.evaluate_with_cv(X_train, y_train)
    for model_name, metrics in cv_results.items():
        print(f"\n{model_name}:")
        for metric_name, metric_result in metrics.items():
            print(f"  {metric_name}: {metric_result['mean']:.3f}±{metric_result['std']:.3f}")

    # LOOCV评估（保留原有功能）
    print("\n=== LOOCV评估结果 ===")
    loocv_results = classifier.evaluate_with_loocv(X_train, y_train)
    for model_name, metrics in loocv_results.items():
        print(f"{model_name}: F1={metrics['mean_f1']:.3f}±{metrics['std_f1']:.3f}")

    # Bootstrap评估（保留原有功能）
    print("\n=== Bootstrap评估结果 ===")
    bootstrap_scores = bootstrap_evaluation(classifier, X_train, y_train)
    print(f"Bootstrap F1: {bootstrap_scores.mean():.3f}±{bootstrap_scores.std():.3f}")
    print(f"95%置信区间: [{np.percentile(bootstrap_scores, 2.5):.3f}, {np.percentile(bootstrap_scores, 97.5):.3f}]")

    return classifier

if __name__ == "__main__":
    classifier = main()
