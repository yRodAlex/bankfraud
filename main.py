import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title='Bank Account Fraud Detection', page_icon='🎫', layout="wide")
st.title('Bank Account Fraud Detection')
st.info('Você pode importar o seus dados para verificar fraude nas contas bancárias')

# Upload do arquivo
with st.sidebar:
    uploaded_file = st.file_uploader("Escolha o Arquivo: ", type=['csv'])
    data = None

    if uploaded_file is not None:
        chunk_size = 10**6
        chunks = []
        try:
            for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
                chunks.append(chunk)
                if len(chunks) * chunk_size >= 2_000_000:
                    st.warning("Arquivo muito grande! Apenas parte dos dados foi carregada.")
                    break
            data = pd.concat(chunks, ignore_index=True)
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")

    st.info('Clique no botão abaixo para processar o arquivo')
    processar = st.button("Processar")

# Processamento e predição
if uploaded_file is not None and processar:
    try:
        st.write("Pré-visualização dos dados importados:")
        st.dataframe(data.head())

        def codificar_dados(df):
            df = df.copy()
            for col in df.select_dtypes(include='object').columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            for col in df.select_dtypes(include='bool').columns:
                df[col] = df[col].astype(int)
            return df

        def tratar_nulos(X):
            imputer = SimpleImputer(strategy='median')
            return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        def preparar_dados(df):
            df = df.dropna(subset=['fraud_bool'])
            X = df.drop(columns=["fraud_bool"])
            y = df["fraud_bool"]
            X = codificar_dados(X)
            X = tratar_nulos(X)
            return X, y

        def balancear_amostras(X, y):
            smote_enn = SMOTEENN(random_state=42)
            return smote_enn.fit_resample(X, y)

        def treinar_modelo(X, y):
            model = XGBClassifier(
                scale_pos_weight=10,
                eval_metric='logloss',
                use_label_encoder=False,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X, y)
            return model

        # Execução
        X, y = preparar_dados(data)
        X_bal, y_bal = balancear_amostras(X, y)
        modelo = treinar_modelo(X_bal, y_bal)

        # Predição
        y_pred = modelo.predict(X)
        data = data.copy()
        data['predicted_fraud'] = y_pred
        st.success("Predição realizada e salva na coluna 'predicted_fraud'.")

        # Resultados
        st.subheader("Resultados da Classificação")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fraudes reais - previstas", ((y == 1) & (y_pred == 1)).sum())
            st.metric("Fraudes reais - não previstas", ((y == 1) & (y_pred == 0)).sum())
        with col2:
            st.metric("Não fraudes - previstas como fraude", ((y == 0) & (y_pred == 1)).sum())
            st.metric("Não fraudes - não previstas", ((y == 0) & (y_pred == 0)).sum())

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        st.write(f"**Acurácia:** {acc*100:.2f}%")
        st.write(f"**Precisão:** {prec*100:.2f}%")
        st.write(f"**Recall (Sensibilidade):** {rec*100:.2f}%")
        st.write(f"**F1-Score:** {f1*100:.2f}%")

        # Matriz de Confusão
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(cm, index=['Não Fraude', 'Fraude'], columns=['Previsto Não Fraude', 'Previsto Fraude'])
        st.write(cm_df)

        # Gráfico de Barras Customizado
        st.subheader("Distribuição de Fraudes Reais vs Previstas")
        counts = pd.DataFrame({
            'Real': pd.Series(y).value_counts().sort_index(),
            'Previsto': pd.Series(y_pred).value_counts().sort_index()
        })
        counts.index = ['Não Fraude', 'Fraude']

        fig, ax = plt.subplots(figsize=(5, 3))
        bar_width = 0.35
        x = np.arange(len(counts))

        ax.bar(x - bar_width/2, counts['Real'], width=bar_width, label='Real', alpha=0.8)
        ax.bar(x + bar_width/2, counts['Previsto'], width=bar_width, label='Previsto', alpha=0.8)

        ax.set_xlabel('Classe', fontsize=10)
        ax.set_ylabel('Contagem', fontsize=10)
        ax.set_title('Fraudes: Reais vs Previstas', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(counts.index, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

        st.pyplot(fig)

        # Download
        st.subheader("Download dos Resultados")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar arquivo com predições",
            data=csv,
            file_name='dados_com_predicoes.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento: {e}")
