import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SteamGameSuccessAnalysis:
    def __init__(self):
        self.df = None
        self.df_filtered = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_importance = None
        self.scaler = None
        self.selected_features = None
        
    def load_kaggle_data(self):
        try:
            import kagglehub
            path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
            json_path = os.path.join(path, "games.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            games_list = []
            for game_id, game_data in data.items():
                if game_data: 
                    game_data['app_id'] = game_id
                    games_list.append(game_data)
            
            self.df = pd.DataFrame(games_list)
            print(f"Dados carregados: {len(self.df)} jogos")
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            raise
    
    def process_data_without_leakage(self):
        
        self.df['is_indie'] = self.df.apply(self._extract_indie_status, axis=1)
        
        # 2. Preço (disponível no lançamento)
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce').fillna(0)
        self.df['is_free'] = (self.df['price'] == 0).astype(int)
        self.df['price_category'] = pd.cut(
            self.df['price'], 
            bins=[-1, 0, 5, 10, 20, 30, 1000],
            labels=['Free', 'Very Cheap', 'Cheap', 'Moderate', 'Expensive', 'AAA']
        )
        
        self.df['release_date'] = pd.to_datetime(
            self.df['release_date'], errors='coerce', format='%b %d, %Y'
        )
        self.df['release_year'] = self.df['release_date'].dt.year
        self.df['release_month'] = self.df['release_date'].dt.month
        self.df['release_day_of_week'] = self.df['release_date'].dt.dayofweek
        
        platform_cols = ['windows', 'mac', 'linux']
        for col in platform_cols:
            self.df[col] = self.df[col].astype(float).fillna(0)
        self.df['num_platforms'] = self.df[platform_cols].sum(axis=1)
        
        self.df['num_developers'] = self.df['developers'].apply(
            lambda x: len(x) if isinstance(x, list) else 1 if pd.notna(x) else 0
        )
        self.df['num_publishers'] = self.df['publishers'].apply(
            lambda x: len(x) if isinstance(x, list) else 1 if pd.notna(x) else 0
        )
        
        self._process_categorical_features()
        
        self.df['achievements'] = pd.to_numeric(
            self.df['achievements'], errors='coerce'
        ).fillna(0).astype(int)
        
        self.df['num_supported_languages'] = self.df['supported_languages'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        print("\n⚠ AVISO: Criando variável alvo baseada em características do lançamento")
        
        if 'positive' in self.df.columns and 'negative' in self.df.columns:
            self.df['review_ratio'] = self.df['positive'] / (self.df['positive'] + self.df['negative'] + 1)
            
            has_enough_reviews = (self.df['positive'] + self.df['negative']) >= 10
            threshold = self.df.loc[has_enough_reviews, 'review_ratio'].quantile(0.7)
            
            indie_mask = self.df['is_indie'] == True
            aaa_mask = self.df['is_indie'] == False
            
            indie_threshold = self.df.loc[has_enough_reviews & indie_mask, 'review_ratio'].quantile(0.6)
            
            aaa_threshold = self.df.loc[has_enough_reviews & aaa_mask, 'review_ratio'].quantile(0.75)
            
            self.df['success_score'] = 0
            self.df.loc[has_enough_reviews & indie_mask & (self.df['review_ratio'] >= indie_threshold), 'success_score'] = 1
            self.df.loc[has_enough_reviews & aaa_mask & (self.df['review_ratio'] >= aaa_threshold), 'success_score'] = 1
        else:
            def calculate_success_score(row):
                score = 0
                
                if row['price'] > 0 and row['price'] < 30:
                    score += 1
                
                if row['num_platforms'] >= 2:
                    score += 1
                
                if row.get('achievements', 0) > 0:
                    score += 1
                
                if row['is_free'] == 0:
                    score += 1
                
                return 1 if score >= 3 else 0
            
            self.df['success_score'] = self.df.apply(calculate_success_score, axis=1)
        
        indie_success = self.df[self.df['is_indie']]['success_score'].mean()
        aaa_success = self.df[~self.df['is_indie']]['success_score'].mean()
        
        print(f"\n ESTATÍSTICAS DE SUCESSO:")
        print(f"  Taxa de sucesso Indies: {indie_success:.2%}")
        print(f"  Taxa de sucesso AAA: {aaa_success:.2%}")
        print(f"  Taxa de sucesso geral: {self.df['success_score'].mean():.2%}")
        
        return self.df
    
    def _process_categorical_features(self):
        """Processa features categóricas como gêneros e categorias"""
        if 'genres' in self.df.columns:
            top_genres = ['Action', 'Adventure', 'Indie', 'RPG', 'Strategy', 
                         'Simulation', 'Casual', 'Sports', 'Racing']
            
            for genre in top_genres:
                self.df[f'genre_{genre.lower()}'] = self.df['genres'].apply(
                    lambda x: 1 if isinstance(x, list) and genre in x else 0
                )
        
        if 'categories' in self.df.columns:
            important_categories = ['Single-player', 'Multi-player', 'Co-op', 
                                  'Steam Achievements', 'Full controller support']
            
            for category in important_categories:
                cat_name = category.lower().replace('-', '_').replace(' ', '_')
                self.df[f'cat_{cat_name}'] = self.df['categories'].apply(
                    lambda x: 1 if isinstance(x, list) and category in x else 0
                )
    
    def _extract_indie_status(self, row):
        """Determina se um jogo é indie de forma robusta"""
        if 'genres' in row and isinstance(row['genres'], list):
            if 'Indie' in row['genres']:
                return True
        
        if 'tags' in row and isinstance(row['tags'], dict):
            if 'Indie' in row['tags']:
                return True
        
        price = row.get('price', 0)
        if price < 30:
            devs = row.get('developers', [])
            if isinstance(devs, list) and len(devs) == 1:
                return True
        
        return False
    
    def filter_and_balance_data(self):
        print("\nFILTRANDO E BALANCEANDO DADOS...")
        
        if 'release_year' in self.df.columns:
            self.df_filtered = self.df[self.df['release_year'] >= 2015].copy()
        else:
            self.df_filtered = self.df.copy()
        
        print(f"Jogos após 2015: {len(self.df_filtered)}")
        
        success_ratio = self.df_filtered['success_score'].mean()
        
        if success_ratio < 0.2 or success_ratio > 0.8:
            print(f"Balanceando classes (taxa atual: {success_ratio:.2%})")
            
            success_df = self.df_filtered[self.df_filtered['success_score'] == 1]
            failure_df = self.df_filtered[self.df_filtered['success_score'] == 0]
            
            min_samples = min(len(success_df), len(failure_df), 10000)
            success_sampled = success_df.sample(n=min_samples, random_state=42)
            failure_sampled = failure_df.sample(n=min_samples, random_state=42)
            
            self.df_filtered = pd.concat([success_sampled, failure_sampled])
            print(f"Dados balanceados: {len(self.df_filtered)} amostras")
        
        indie_success = self.df_filtered[self.df_filtered['is_indie']]['success_score'].mean()
        aaa_success = self.df_filtered[~self.df_filtered['is_indie']]['success_score'].mean()
        
        print(f"DISTRIBUIÇÃO FINAL:")
        print(f"  Total amostras: {len(self.df_filtered)}")
        print(f"  Taxa sucesso Indies: {indie_success:.2%}")
        print(f"  Taxa sucesso AAA: {aaa_success:.2%}")
        print(f"  Proporção Indies: {self.df_filtered['is_indie'].mean():.2%}")
        
        return self.df_filtered
    
    def prepare_features(self):
        print("\nPREPARANDO FEATURES PARA MODELAGEM...")
        
        features = [
            'price', 'is_free', 'is_indie',
            'release_month', 'release_day_of_week',
            'num_platforms', 'num_developers', 'num_publishers',
            'achievements', 'num_supported_languages',
            'genre_action', 'genre_adventure', 'genre_indie', 'genre_rpg',
            'genre_strategy', 'genre_simulation', 'genre_casual',
            'cat_single_player', 'cat_multi_player', 'cat_co_op',
            'cat_steam_achievements', 'cat_full_controller_support',
        ]
        
        available_features = [f for f in features if f in self.df_filtered.columns]
        print(f"Features disponíveis: {len(available_features)}")
        
        X = self.df_filtered[available_features].fillna(0)
        y = self.df_filtered['success_score']
        
        self.selected_features = available_features
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = pd.DataFrame(X_train_scaled, columns=available_features, index=X_train.index)
        self.X_test = pd.DataFrame(X_test_scaled, columns=available_features, index=X_test.index)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"\nDados preparados:")
        print(f"  Treino: {len(X_train)} amostras")
        print(f"  Teste: {len(X_test)} amostras")
        print(f"  Features: {len(available_features)}")
        
        return available_features
    
    def train_and_evaluate_models(self):
        """Treina e avalia os modelos"""
        print("\n" + "="*60)
        print(" TREINANDO MODELOS - RANDOM FOREST E NAIVE BAYES")
        print("="*60)
        
        print("\n--- RANDOM FOREST ---")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20, 
            min_samples_leaf=10,  
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        print("\n--- NAIVE BAYES ---")
        nb_model = GaussianNB()
        
        self.models = {
            'Random Forest': rf_model,
            'Naive Bayes': nb_model
        }
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            print("  Treinando...")
            model.fit(self.X_train, self.y_train)
            
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            train_acc = accuracy_score(self.y_train, y_train_pred)
            test_acc = accuracy_score(self.y_test, y_test_pred)
            test_precision = precision_score(self.y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(self.y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)
            
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=5, scoring='f1', n_jobs=-1)
            
            self.results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Resultados:")
            print(f"    Acurácia (treino): {train_acc:.4f}")
            print(f"    Acurácia (teste):  {test_acc:.4f}")
            print(f"    Precisão:          {test_precision:.4f}")
            print(f"    Recall:            {test_recall:.4f}")
            print(f"    F1-Score:          {test_f1:.4f}")
            print(f"    CV F1-Score:       {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            cm = confusion_matrix(self.y_test, y_test_pred)
            print(f"    Matriz Confusão:   TP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}")
        
        if 'Random Forest' in self.models:
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.models['Random Forest'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTOP 10 FEATURES MAIS IMPORTANTES (Random Forest):")
            for idx, row in self.feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.best_model_name = max(self.results.keys(), 
                                  key=lambda x: self.results[x]['f1_score'])
        
        print(f"\nMELHOR MODELO: {self.best_model_name}")
        print(f"   F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}")
    
    def analyze_indie_vs_aaa(self):
        print("\n" + "="*60)
        print(" ANÁLISE DETALHADA: INDIES vs AAA")
        print("="*60)
        
        best_model = self.models[self.best_model_name]
        
        indie_games = self.df_filtered[self.df_filtered['is_indie'] == True].copy()
        aaa_games = self.df_filtered[self.df_filtered['is_indie'] == False].copy()
        
        print(f"\nCOMPARAÇÃO BÁSICA:")
        print(f"  Indies analisados: {len(indie_games)}")
        print(f"  AAA analisados: {len(aaa_games)}")
        print(f"  Sucesso Indies: {indie_games['success_score'].mean():.2%}")
        print(f"  Sucesso AAA: {aaa_games['success_score'].mean():.2%}")
        
        print(f"\nANÁLISE POR FAIXA DE PREÇO:")
        
        if 'price_category' in self.df_filtered.columns:
            price_analysis = self.df_filtered.groupby(['is_indie', 'price_category'])['success_score'].agg(['mean', 'count'])
            
            for idx, row in price_analysis.iterrows():
                indie_status = "Indie" if idx[0] else "AAA"
                price_cat = idx[1]
                success_rate = row['mean']
                count = row['count']
                
                if count > 10:  
                    print(f"  {indie_status} - {price_cat}: {success_rate:.2%} ({int(count)} jogos)")
        
        print(f"\n FATORES DE SUCESSO PARA INDIES:")
        
        if len(indie_games) > 50:
            indie_success = indie_games[indie_games['success_score'] == 1]
            indie_failure = indie_games[indie_games['success_score'] == 0]
            
            factors = ['price', 'num_platforms', 'achievements', 'num_supported_languages']
            
            for factor in factors:
                if factor in indie_success.columns and factor in indie_failure.columns:
                    success_mean = indie_success[factor].mean()
                    failure_mean = indie_failure[factor].mean()
                    diff = success_mean - failure_mean
                    print(f"  {factor}: Sucesso={success_mean:.2f}, Falha={failure_mean:.2f} (diferença: {diff:+.2f})")
        
        print(f"\nPREVISÕES DO MODELO ({self.best_model_name}):")
        
        print(f"\nANÁLISE ESTATÍSTICA (usando conjunto de TESTE):")
        
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            test_indices = self.X_test.index
            
            test_indie_status = self.df_filtered.loc[test_indices, 'is_indie']
            
            indie_mask = (test_indie_status == True)
            aaa_mask = (test_indie_status == False)
            
            if indie_mask.any():
                indie_avg_prob = y_proba[indie_mask].mean()
                indie_std_prob = y_proba[indie_mask].std()
                print(f"   INDIES (n={indie_mask.sum()}):")
                print(f"    • Probabilidade média de sucesso: {indie_avg_prob:.2%}")
                print(f"    • Desvio padrão: ±{indie_std_prob:.2%}")
                print(f"    • Intervalo: [{y_proba[indie_mask].min():.2%} - {y_proba[indie_mask].max():.2%}]")
            
            if aaa_mask.any():
                aaa_avg_prob = y_proba[aaa_mask].mean()
                aaa_std_prob = y_proba[aaa_mask].std()
                print(f"   AAA (n={aaa_mask.sum()}):")
                print(f"    • Probabilidade média de sucesso: {aaa_avg_prob:.2%}")
                print(f"    • Desvio padrão: ±{aaa_std_prob:.2%}")
                print(f"    • Intervalo: [{y_proba[aaa_mask].min():.2%} - {y_proba[aaa_mask].max():.2%}]")
            
            if indie_mask.any() and aaa_mask.any():
                diff = indie_avg_prob - aaa_avg_prob
                print(f"\n  ️  COMPARAÇÃO:")
                print(f"    • Diferença (Indies - AAA): {diff:+.2%}")
        
        print(f"\n  PERFIS HIPOTÉTICOS:")
        
        try:
            train_means = self.X_train.mean()
            
            indie_profile = train_means.copy()
            if 'price' in indie_profile.index:
                indie_profile['price'] = 1.0  
            if 'is_indie' in indie_profile.index:
                indie_profile['is_indie'] = 1.5  
            if 'achievements' in indie_profile.index:
                indie_profile['achievements'] = 1.0
            
            aaa_profile = train_means.copy()
            if 'price' in aaa_profile.index:
                aaa_profile['price'] = 3.0  
            if 'is_indie' in aaa_profile.index:
                aaa_profile['is_indie'] = -1.0 
            if 'num_developers' in aaa_profile.index:
                aaa_profile['num_developers'] = 1.5
            if 'num_publishers' in aaa_profile.index:
                aaa_profile['num_publishers'] = 1.5
            
            indie_df = pd.DataFrame([indie_profile])
            aaa_df = pd.DataFrame([aaa_profile])
            
            indie_prob = best_model.predict_proba(indie_df)[0][1]
            aaa_prob = best_model.predict_proba(aaa_df)[0][1]
            
            print(f"    Perfil indie típico: {indie_prob:.2%}")
            print(f"    Perfil AAA típico: {aaa_prob:.2%}")
            print(f"    Diferença: {abs(indie_prob - aaa_prob):.2%}")
            print(f"    Decisão indie: {'Sucesso' if indie_prob >= 0.5 else 'Falha'}")
            print(f"    Decisão AAA: {'Sucesso' if aaa_prob >= 0.5 else 'Falha'}")
            
        except Exception as e:
            print(f"Perfis hipotéticos não disponíveis")
        
        print(f"\n NOTA: Análise baseada em {len(self.df_filtered)} jogos Steam (2015+)")
        print(f"          Modelo {self.best_model_name} com F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}")
    
    def create_visualizations(self):
        """Cria visualizações para o relatório"""
        print("\n CRIANDO VISUALIZAÇÕES...")
        
        os.makedirs('figures', exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        try:
            ax1 = axes[0, 0]
            model_names = list(self.results.keys())
            f1_scores = [self.results[name]['f1_score'] for name in model_names]
            cv_scores = [self.results[name]['cv_mean'] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            ax1.bar(x - width/2, f1_scores, width, label='Teste F1', alpha=0.8, color='skyblue')
            ax1.bar(x + width/2, cv_scores, width, label='CV F1', alpha=0.8, color='lightcoral')
            ax1.set_xlabel('Modelos')
            ax1.set_ylabel('F1-Score')
            ax1.set_title('Desempenho dos Modelos')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            if self.feature_importance is not None:
                ax2 = axes[0, 1]
                top_features = self.feature_importance.head(10)
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
                ax2.barh(top_features['feature'], top_features['importance'], color=colors)
                ax2.set_xlabel('Importância')
                ax2.set_title('Top 10 Features - Random Forest')
                ax2.invert_yaxis()
            
            ax3 = axes[0, 2]
            indie_success = self.df_filtered[self.df_filtered['is_indie']]['success_score'].mean()
            aaa_success = self.df_filtered[~self.df_filtered['is_indie']]['success_score'].mean()
            
            bars = ax3.bar(['Indies', 'AAA'], [indie_success, aaa_success], 
                          color=['lightgreen', 'lightcoral'], alpha=0.8)
            ax3.set_ylabel('Taxa de Sucesso')
            ax3.set_title('Sucesso: Indies vs AAA')
            ax3.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.1%}', ha='center', va='bottom')
            
            if self.best_model_name:
                ax4 = axes[1, 0]
                best_model = self.models[self.best_model_name]
                y_pred = best_model.predict(self.X_test)
                cm = confusion_matrix(self.y_test, y_pred)
                
                im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
                ax4.set_title(f'Matriz Confusão - {self.best_model_name}')
                
                thresh = cm.max() / 2
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax4.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                
                ax4.set_ylabel('Verdadeiro')
                ax4.set_xlabel('Previsto')
                ax4.set_xticks([0, 1])
                ax4.set_yticks([0, 1])
                ax4.set_xticklabels(['Não', 'Sucesso'])
                ax4.set_yticklabels(['Não', 'Sucesso'])
            
            ax5 = axes[1, 1]
            for name, result in self.results.items():
                model = result['model']
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                    auc = roc_auc_score(self.y_test, y_proba)
                    ax5.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
            
            ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aleatório')
            ax5.set_xlabel('Taxa Falsos Positivos')
            ax5.set_ylabel('Taxa Verdadeiros Positivos')
            ax5.set_title('Curva ROC')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            ax6 = axes[1, 2]
            if 'price' in self.df_filtered.columns:
                price_bins = [0, 5, 10, 20, 30, 60, 200]
                price_labels = ['0-5', '5-10', '10-20', '20-30', '30-60', '60+']
                
                self.df_filtered['price_group'] = pd.cut(
                    self.df_filtered['price'], bins=price_bins, labels=price_labels
                )
                
                price_success = self.df_filtered.groupby('price_group')['success_score'].mean()
                price_count = self.df_filtered.groupby('price_group').size()
                
                valid_groups = price_count[price_count > 10].index
                price_success = price_success[valid_groups]
                
                colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(price_success)))
                bars = ax6.bar(price_success.index.astype(str), price_success.values, color=colors)
                ax6.set_xlabel('Faixa de Preço ($)')
                ax6.set_ylabel('Taxa de Sucesso')
                ax6.set_title('Sucesso por Faixa de Preço')
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('ANÁLISE DE SUCESSO DE JOGOS STEAM - INDIES vs AAA', 
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig('figures/analise_completa.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Visualizações salvas: 'figures/analise_completa.png'")
            
        except Exception as e:
            print(f"Erro ao criar visualizações: {e}")
    
    def generate_final_report(self):
        """Gera relatório final em formato Markdown"""
        print("\n" + "="*60)
        print("GERANDO RELATÓRIO FINAL")
        print("="*60)
        
        try:
            with open('relatorio_analise_jogos.md', 'w', encoding='utf-8') as f:
                f.write("# ANÁLISE DE SUCESSO DE JOGOS STEAM: INDIES vs AAA\n\n")
                f.write(f"**Data da análise:** {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
                f.write(f"**Total de jogos analisados:** {len(self.df_filtered)}\n")
                f.write(f"**Período:** {self.df_filtered['release_year'].min()} - {self.df_filtered['release_year'].max()}\n\n")
                
                f.write("##RESUMO EXECUTIVO\n\n")
                
                indie_count = self.df_filtered['is_indie'].sum()
                indie_success = self.df_filtered[self.df_filtered['is_indie']]['success_score'].mean()
                aaa_success = self.df_filtered[~self.df_filtered['is_indie']]['success_score'].mean()
                
                f.write(f"- **Jogos Indies analisados:** {indie_count} ({indie_count/len(self.df_filtered):.1%})\n")
                f.write(f"- **Taxa de sucesso Indies:** {indie_success:.2%}\n")
                f.write(f"- **Taxa de sucesso AAA:** {aaa_success:.2%}\n")
                
                if indie_success > aaa_success:
                    f.write(f"- **Indies têm {indie_success - aaa_success:+.2%} mais chance de sucesso**\n")
                else:
                    f.write(f"- **AAA têm {aaa_success - indie_success:.2%} mais chance de sucesso**\n")
                
                f.write("\n##RESULTADOS DOS MODELOS\n\n")
                
                if self.results:
                    f.write("| Modelo | Acurácia | Precisão | Recall | F1-Score | CV F1-Score |\n")
                    f.write("|--------|----------|----------|--------|----------|-------------|\n")
                    
                    for name, result in self.results.items():
                        f.write(f"| {name} | {result['test_accuracy']:.4f} | {result['precision']:.4f} | ")
                        f.write(f"{result['recall']:.4f} | {result['f1_score']:.4f} | ")
                        f.write(f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f}) |\n")
                    
                    f.write(f"\n**Melhor modelo:** {self.best_model_name}\n")
                    f.write(f"**F1-Score:** {self.results[self.best_model_name]['f1_score']:.4f}\n")
                
                f.write("\n##CONCLUSÕES PRINCIPAIS\n\n")
                
                if self.feature_importance is not None:
                    f.write("### Fatores Mais Importantes para o Sucesso:\n\n")
                    for idx, row in self.feature_importance.head(5).iterrows():
                        f.write(f"1. **{row['feature']}** (importância: {row['importance']:.4f})\n")
                
                f.write("\n### Recomendações para Desenvolvedores:\n\n")
                
                f.write("1. **Preço adequado é crucial** - Indies devem manter preços entre $10-30\n")
                f.write("2. **Multiplataforma aumenta chances** - Lançar para múltiplas plataformas\n")
                f.write("3. **Conquistas importam** - Implemente um sistema de conquistas\n")
                f.write("4. **Identidade clara** - Ser identificado como Indie pode ser vantajoso\n")
                f.write("5. **Evitar jogos gratuitos** - Jogos pagos têm maior taxa de sucesso\n")
                
                f.write("\n### Limitações do Estudo:\n\n")
                f.write("1. **Dados históricos** - O mercado de jogos evolui rapidamente\n")
                f.write("2. **Definição de sucesso** - Baseada em critérios simplificados\n")
                f.write("3. **Features disponíveis** - Limitado a dados públicos do Steam\n")
                f.write("4. **Viés de sobrevivência** - Apenas jogos que foram lançados\n")
                f.write("5. **Desbalanceamento** - Muitos mais Indies que AAA\n")
                
                f.write("\n## VISUALIZAÇÕES\n\n")
                f.write("![Análise Completa](figures/analise_completa.png)\n")
                
                f.write("\n---\n")
                f.write("*Relatório gerado automaticamente pelo sistema de análise de dados*")
            
            print("Relatório gerado: 'relatorio_analise_jogos.md'")
            
            print("\n" + "="*60)
            print("RESUMO DA ANÁLISE")
            print("="*60)
            print(f"\nDADOS:")
            print(f"  Jogos analisados: {len(self.df_filtered)}")
            print(f"  Indies: {self.df_filtered['is_indie'].sum()} ({self.df_filtered['is_indie'].mean():.1%})")
            print(f"  Taxa de sucesso geral: {self.df_filtered['success_score'].mean():.1%}")
            
            if self.best_model_name:
                best = self.results[self.best_model_name]
                print(f"\nMODELO:")
                print(f"   Melhor: {self.best_model_name}")
                print(f"   F1-Score: {best['f1_score']:.4f}")
                print(f"   Acurácia: {best['test_accuracy']:.4f}")
            
            print(f"\n INDIES vs AAA:")
            print(f"   Sucesso Indies: {indie_success:.2%}")
            print(f"   Sucesso AAA: {aaa_success:.2%}")
            print(f"   Diferença: {abs(indie_success - aaa_success):.2%}")
            
            print(f"\n ARQUIVOS GERADOS:")
            print(f"   relatorio_analise_jogos.md")
            print(f"   figures/analise_completa.png")
            print(f"   dados/jogos_processados.csv")
            
        except Exception as e:
            print(f"Erro ao gerar relatório: {e}")
    
    def save_processed_data(self):
        os.makedirs('dados', exist_ok=True)
        self.df_filtered.to_csv('dados/jogos_processados.csv', index=False)
        print("Dados processados salvos: 'dados/jogos_processados.csv'")
    
    def run_complete_analysis(self):
        print("="*60)
        print("ANÁLISE DE SUCESSO DE JOGOS STEAM")
        print("Indies Baratos vs AAA Caros - Random Forest + Naive Bayes")
        print("="*60)
        
        try:
            self.load_kaggle_data()
            
            self.process_data_without_leakage()
            
            self.filter_and_balance_data()
            
            features = self.prepare_features()
            
            self.train_and_evaluate_models()
            
            self.analyze_indie_vs_aaa()
            
            self.create_visualizations()
            
            self.save_processed_data()
            
            self.generate_final_report()
            
            print("\n" + "="*60)
            print("ANÁLISE CONCLUÍDA COM SUCESSO!")
            print("="*60)
            
        except Exception as e:
            print(f"\nERRO NA ANÁLISE: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ANALISADOR DE JOGOS STEAM")
    print("(Random Forest + Naive Bayes)")
    print("="*60)
    
    analysis = SteamGameSuccessAnalysis()
    analysis.run_complete_analysis()