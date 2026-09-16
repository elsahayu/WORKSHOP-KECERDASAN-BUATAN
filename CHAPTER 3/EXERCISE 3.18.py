import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from auto_ml import Predictor

# 1. Load dataset California Housing
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['target'] = california.target

# 2. Bagi dataset menjadi Train dan Test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# 3. Deskripsi kolom target
column_descriptions = {
    'target': 'output'
}

# 4. Inisialisasi dan jalankan AutoML
ml_predictor = Predictor(type_of_estimator='regressor', 
                         column_descriptions=column_descriptions)

ml_predictor.train(df_train)
ml_predictor.score(df_test, df_test.target)