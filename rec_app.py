# Streamlit Code
import pickle
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Define the Recommender Class
class BookRecommender:
    def __init__(self, df_mf, book_info, book_rec, ml_train_set, gbr_model, svd_model):
        self.df_mf = df_mf
        self.book_info = book_info
        self.book_rec = book_rec
        self.ml_train_set = ml_train_set
        self.gbr_model = gbr_model
        self.svd_model = svd_model
        self.cosine_sim = None
        self.create_cosine_matrix()

    def create_cosine_matrix(self):
        """ Create the cosine similarity matrix for content-based filtering """
        tfidf = TfidfVectorizer(stop_words='english', max_features=100, min_df=10)
        tfidf_matrix = tfidf.fit_transform(self.book_rec['keywords'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_recommended_items_mf(self, user_id, top_n):
        """ Matrix Factorization based recommendations """
        user_reviewed_iid = self.df_mf[self.df_mf['user_id'] == user_id]['isbn']
        iid_to_est = self.df_mf[~self.df_mf['isbn'].isin(user_reviewed_iid)]['isbn'].drop_duplicates()
        testset = [(user_id, iid, None) for iid in iid_to_est]
        
        predictions = self.svd_model.test(testset)
        predictions = pd.DataFrame(predictions)[['uid', 'iid', 'est']]
        predictions.columns = ['user_id', 'isbn', 'predicted_rating']
        predictions['predicted_rank'] = predictions['predicted_rating'].rank(method='first', ascending=False)
        predictions = predictions.sort_values("predicted_rank")[:top_n]
        predictions = predictions.merge(self.book_info, on='isbn')
        
        return predictions[['user_id', 'isbn', 'book_title', 'book_author']]

    def get_recommended_items_content(self, user_id, top_n):
        """ Content-based recommendations """
        most_liked_isbn = self.df_mf[self.df_mf['user_id'] == user_id].sort_values("rating", ascending=False)['isbn']
        if most_liked_isbn.empty:
            return pd.DataFrame()

        most_liked_isbn = most_liked_isbn.values[0]
        
        if most_liked_isbn not in self.book_rec['isbn'].values:
            predictions = book_info.sort_values("rating_count", ascending=False)[:top_n]
            predictions['user_id'] = user_id
            predictions = predictions[['user_id', 'isbn', 'book_title', 'book_author']]
        else:
        
            idx = self.book_rec[self.book_rec['isbn'] == most_liked_isbn].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            book_indices = [i[0] for i in sim_scores]
            recommend_books = self.book_rec.iloc[book_indices]['isbn']
            predictions = self.book_info[self.book_info['isbn'].isin(recommend_books)]
            predictions['user_id'] = user_id
            predictions = predictions[['user_id', 'isbn', 'book_title', 'book_author']]

        return predictions

    def get_recommended_items_ml(self, user_id, top_n):
        """ Machine Learning (GBR) based recommendations """
        user_reviewed_iid = self.df_mf[self.df_mf['user_id'] == user_id]['isbn']
        iid_to_est = self.df_mf[~self.df_mf['isbn'].isin(user_reviewed_iid)]['isbn'].drop_duplicates()
        ml_test_set = self.ml_train_set[self.ml_train_set['isbn'].isin(iid_to_est)]
        user_age = self.ml_train_set[self.ml_train_set['user_id'] == user_id]['age'].unique()

        features = ['age', 'isbn_encoded', 'book_author_encoded', 'year_of_publication',
                    'rating_mean', 'rating_count', 'category_encoded']
        
        X = ml_test_set[features].query("age == @user_age")
        y_pred = self.gbr_model.predict(X)
        
        gbr_rec_result = X.copy()
        gbr_rec_result['y_pred'] = y_pred
        gbr_rec_result = gbr_rec_result.sort_values("y_pred", ascending=False)[:top_n]
        gbr_rec_result = gbr_rec_result.merge(self.ml_train_set[['isbn_encoded', 'isbn', 'book_title', 'book_author']].drop_duplicates(), on='isbn_encoded')
        gbr_rec_result = gbr_rec_result[['isbn', 'book_title', 'book_author']]
        gbr_rec_result['user_id'] = user_id
        gbr_rec_result = gbr_rec_result.reindex(columns=['user_id', 'isbn', 'book_title', 'book_author'])

        return gbr_rec_result

with st.spinner('결과를 열심히 가져오고 있어요. 잠시만 기다려주세요.'): 

    # Define your file paths here
    df_mf_path = 'data/df_mf.csv'
    book_info_path = 'data/book_info.csv'
    user_info_path = 'data/user_info.csv'
    book_rec_path = 'data/book_rec.csv'
    ml_train_set_path = 'data/ml_train_set.csv'

    # Load data and models
    @st.cache_data
    def load_data():
        df_mf = pd.read_csv(df_mf_path)
        book_info = pd.read_csv(book_info_path)
        user_info = pd.read_csv(user_info_path)
        book_rec = pd.read_csv(book_rec_path)
        ml_train_set = pd.read_csv(ml_train_set_path)

        return df_mf, book_info, user_info, book_rec, ml_train_set

    df_mf, book_info, user_info, book_rec, ml_train_set = load_data()

    @st.cache_resource
    def load_models():
        with open('model/gbr_trained_model.pkl', 'rb') as file:
            gbr = pickle.load(file)

        with open('model/svd_trained_model.pkl', 'rb') as file:
            svd = pickle.load(file)

        return gbr, svd

    gbr_model, svd_model = load_models()

    # Initialize the Recommender System
    book_recommender = BookRecommender(df_mf, book_info, book_rec, ml_train_set, gbr_model, svd_model)

    # Sidebar for user inputs
    st.sidebar.header("Input User Info.")
    unique_user_ids = ml_train_set['user_id'].unique()
    user_id = st.sidebar.selectbox("Select User ID", unique_user_ids)
    top_n = st.sidebar.slider("Select the number of recommendations:", min_value=1, max_value=20, value=10)    

    # Frontend Display using Tabs
    st.title("📚 Book Recommender System")
    st.markdown("이 어플리케이션은 3개의 알고리즘을 이용하여 책을 추천합니다.")

    # Define tabs
    tab1, tab2, tab3 = st.tabs(["Matrix Factorization", "Content-based", "GBR Model"])

    # Matrix Factorization Tab
    with tab1:
        st.markdown("**✅ Matrix Factorization Recommendations**")
        st.info("Matrix Factorization 은 유저 및 책의 잠재요인을 파악하여 추천하는 알고리즘입니다.")
        mf_rec_result = book_recommender.get_recommended_items_mf(user_id, top_n)
        st.table(mf_rec_result)

    # Content-based Tab
    with tab2:
        st.markdown("**✅ Content-based Recommendations**")
        st.info("유저의 최근 선호 아이템과 유사한 아이템을 추천합니다.")
        content_rec_result = book_recommender.get_recommended_items_content(user_id, top_n)
        st.table(content_rec_result)

    # GBR Model Tab
    with tab3:
        st.markdown("**✅ GBR Model Recommendations**")
        st.info("GBR 알고리즘은 머신러닝 예측모델을 이용하여 유저가 좋아할만한 책을 추천합니다.")
        gbr_rec_result = book_recommender.get_recommended_items_ml(user_id, top_n)
        st.table(gbr_rec_result)