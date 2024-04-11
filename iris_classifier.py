from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.metrics import classification_report, accuracy_score

# データ取得
iris = load_iris()
x, y = iris.data, iris.target

# 訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

# solverには確率的勾配降下法(sgd)やadamなどが利用可能です。
model = MLPClassifier(solver="sgd", random_state=0, max_iter=3000)

# 学習
model.fit(x_train, y_train)
pred = model.predict(x_test)

# 学習済みモデルの保存
joblib.dump(model, "nn.pkl", compress=True)


# 精度と近藤行列の出力
print("result: ", model.score(x_test, y_test))
print(classification_report(y_test, pred))

from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
import gradio as gr
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def predict(sepal_length, sepal_width,
            petal_length, petal_width):
    mat = np.array([sepal_length, sepal_width,
                    petal_length, petal_width])
    mat = mat.reshape(1, mat.shape[0])
    df = pd.DataFrame(mat, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    # res = knn.predict(df)
    # return res[0]
    knn = KNeighborsClassifier(n_neighbors=6)
    res = knn.predict_proba(df)
    res_dict = {}
    for i in range(len(res[0])):
        # NOTE: クラスと確率の組をdictで返すとラベルに両方とも綺麗に表示される。
        # そのときkeyにするクラスの型は文字列にするのが望ましい。
        res_dict[str(i)] = res[0][i]
    return res_dict


# ラベルからIrisの名前を取得します
def getName(label):
    print(label)
    if label == 0:
        return "Iris Setosa"
    elif label == 1: 
        return "Iris Versicolor"
    elif label == 2: 
        return "Iris Virginica"
    else: 
        return "Error"

def recognition_flower(sepal_length, sepal_width, petal_length, petal_width):
    x = np.array([int(sepal_length), int(sepal_width), int(petal_length), int(petal_width)])
    pred = predict(x)
    irisName = getName(pred)
    return irisName


# Define the interface
sepal_length = gr.inputs.Slider(
    minimum=1, maximum=10, default=X['sepal length (cm)'].mean(), label='sepal_length')
sepal_width = gr.inputs.Slider(
    minimum=1, maximum=10, default=X['sepal width (cm)'].mean(), label='sepal_width')
petal_length = gr.inputs.Slider(
    minimum=1, maximum=10, default=X['petal length (cm)'].mean(), label='petal_length')
petal_width = gr.inputs.Slider(
    minimum=1, maximum=10, default=X['petal width (cm)'].mean(), label='petal_width')
output_placeholder = gr.outputs.Label()


interface = gr.Interface(predict,
                         [sepal_length, sepal_width,
                          petal_length, petal_width],
                         output_placeholder,
                         examples='flagged/')


interface.launch()