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

# 学習済みモデルを読み込み利用します
def predict(parameters):
    # ニューラルネットワークのモデルを読み込み
    model = joblib.load('./nn.pkl')
    params = parameters.reshape(1, -1)
    pred = model.predict(params)
    return pred

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


demo = gr.Interface(fn=recognition_flower,
                    inputs=[gr.Textbox(label="SepalLength"),
                            gr.Textbox(label="SepalWidth"),
                            gr.Textbox(label="PetalLength"),
                            gr.Textbox(label="PetalWidth")
                    ],
                    outputs="text")
demo.launch()