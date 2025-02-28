from flask import Flask, render_template
from kmeans_iris import kmeans_clustering, 
plot_2d_scatter, plot_3d_scatter

app = Flask(__name__)

@app.route('/')
def home():
    df, target_names = kmeans_clustering()
    plot_2d_scatter(df)
    plot_3d_scatter(df)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

