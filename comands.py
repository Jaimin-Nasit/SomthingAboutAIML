{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "848e9039-8c55-41eb-9fa8-ac28a64c8228",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.datasets import load_iris\n",
    "import plotly.express as px"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "d82c338f-fcd3-4668-b74c-955a8eb373b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n",
       "0                5.1               3.5                1.4               0.2\n",
       "1                4.9               3.0                1.4               0.2\n",
       "2                4.7               3.2                1.3               0.2\n",
       "3                4.6               3.1                1.5               0.2\n",
       "4                5.0               3.6                1.4               0.2"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris=load_iris()\n",
    "df=pd.DataFrame(iris.data,columns=iris.feature_names)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "373787dc-9e00-43fc-b866-563c3beb92b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def kmeans_clustering():\n",
    "    iris=load_iris()\n",
    "    df=pd.DataFrame(iris.data,columns=iris.feature_names)\n",
    "    scaler=StandardScaler()\n",
    "    df_scaled=scaler.fit_transform(df)\n",
    "    kmeans=KMeans(n_clusters=3,random_state=42,n_init=18)\n",
    "    df['Cluster']=kmeans.fit_predict(df_scaled)\n",
    "    return df,iris.target_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "1efec3b1-eed9-47d8-871e-5b69e72677cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_2d_scatter(df):\n",
    "    plt.figure (figsize=(8,6))\n",
    "    sns.scatterplot(x=df['sepal length (cm)'],\n",
    "        y=df['sepal width (cm)'], hue=df['Cluster'], palette='viridis')\n",
    "    plt.xlabel(\"sepal length(cm)\")\n",
    "    plt.ylabel(\"sepal width (cm)\")\n",
    "    plt.title(\"K-Means Clustering9(2D view)\")\n",
    "    plt.savefig(\"static/plot_2d.png\")\n",
    "    plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "d2934708-6416-4836-b53f-5f3ea7285783",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_3d_scatter(df):\n",
    "    fig=px.scatter_3d(df,x='sepal length (cm)',\n",
    "                      y='sepal width (cm)',\n",
    "                      z='petal length (cm)',\n",
    "                      color=df['Cluster'].astype(str),\n",
    "                      title=\"K-Means Clustering(3D ciew)\")\n",
    "    fig.write_html(\"static/plot_3d.html\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "5078fbdb-dd7b-4fdf-836a-fbbeb65c3a7c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Jaimin Nasit\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1429: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "df,sample=kmeans_clustering()\n",
    "plot_2d_scatter(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "bb238a59-6ad0-4a16-925b-e7b446ea408e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Jaimin Nasit\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1429: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "df,sample=kmeans_clustering()\n",
    "plot_3d_scatter(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "81ef13d4-5883-4e0c-985a-72145f6a9397",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
