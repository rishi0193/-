import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# Load your DataFrame (replace with your data loading logic)
@st.cache_data
def load_data():
    # Replace with your actual df loading logic
    # Example: pd.read_csv('your_file.csv')
    return pd.read_csv("your_audience_data.csv")  # Replace this

df = load_data()
df = df.dropna(subset=['display_name', 'country'])

# Streamlit UI
st.title("🔍 Audience Recommender (Multilingual + Performance Based)")

# Sidebar filters
available_metrics = ['conversions', 'ctr', 'clicks', 'impressions', 'cost_per_conversion','conversions_from_interactions_rate']
available_countries = sorted(df['country'].dropna().unique().tolist())

performance_metric = st.selectbox("📊 Choose Performance Metric", available_metrics)
country_filter_list = st.multiselect("🌍 Choose Countries", available_countries, default=available_countries[:3])
user_keyword = st.text_input("💡 Enter a Keyword/Topic", "download")
top_n = st.slider("🔝 Number of Recommendations", 1, 20, 5)

if st.button("Generate Recommendations"):
    
    def recommend_keywords(user_keyword, performance_metric, country_filter_list, top_n=5, weight_performance=0.6, weight_similarity=0.4):
        if not country_filter_list or 'ALL' in country_filter_list or 'MULTIPLE' in country_filter_list:
            df_filtered = df.copy()
        else:
            df_filtered = df[df['country'].isin(country_filter_list)]

        df_filtered = df_filtered.dropna(subset=[performance_metric])

        group_cols = ['display_name'] #if not country_filter_list and 'MULTIPLE' in country_filter_list else ['display_name', 'country']
        df_grouped = df_filtered.groupby(group_cols, as_index=False).agg({
            'conversions': 'sum',
            'ctr': 'mean',
            'clicks': 'sum',
            'impressions': 'sum',
            'cost_per_conversion': 'mean',
            'conversions_from_interactions_rate': 'mean'
        })

        user_embedding = model.encode(user_keyword, convert_to_tensor=True)
        keyword_list = df_grouped['display_name'].tolist()
        keyword_embeddings = model.encode(keyword_list, convert_to_tensor=True)
        similarities = util.cos_sim(user_embedding, keyword_embeddings)[0]
        df_grouped['similarity'] = similarities.cpu().numpy()
        df_grouped = df_grouped[df_grouped['similarity'] > 0.5]

        scaler = MinMaxScaler()
        df_grouped[['performance_norm', 'similarity_norm']] = scaler.fit_transform(
            df_grouped[[performance_metric, 'similarity']])
        
        df_grouped = df_grouped[df_grouped['performance_norm'] > 0.01]

        df_grouped['combined_score'] = (weight_performance * df_grouped['performance_norm'] +
                                        weight_similarity * df_grouped['similarity_norm'])

        df_grouped = df_grouped.sort_values(by='combined_score', ascending=False)

        display_cols = ['display_name', performance_metric, 'similarity', 'combined_score']
        #if country_filter_list and 'ALL' not in country_filter_list and 'MULTIPLE' not in country_filter_list:
        #    display_cols.append('country')

        return df_grouped[display_cols].head(top_n)

    results = recommend_keywords(user_keyword, performance_metric, country_filter_list, top_n)
    results['similarity'] = results['similarity'].round(4)
    results[performance_metric] = results[performance_metric].astype("float64")
    
    st.subheader("✅ Top Audience Suggestions")
    st.dataframe(results)
