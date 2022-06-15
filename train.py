import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import matplotlib.pyplot as plt
import data_load
import model

from tqdm import tqdm  #コマンドラインで実行するとき
# from tqdm.notebook import tqdm  # jupyter で実行するとき

# リソースの指定（CPU/GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# 訓練の実行
epoch = 100
train_loss = []
test_loss = []

for epoch in tqdm(range(epoch)):
    model, train_l, test_l = train_model(model)
    train_loss.append(train_l)
    test_loss.append(test_loss)
    # 10エポックごとにロスを表示
    if epoch % 10 == 0:
        print("Train loss: {a:.3f}, Test loss: {b:.3f}".format(a=train_loss[-1], b = test_loss[-1]))

# 学習状況（ロス）の確認
plt.plot(train_loss, label='train_loss')
plt.plot(test_loss, label='test_loss')
plt.legend()








# 学習済みモデルから予測結果と正解値を取得
def retrieve_result(model, dataloader):
    model.eval()
    preds = []
    labels = []
    # Retreive prediction and labels
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            # Collect data
            preds.append(output)
            labels.append(label)
    # Flatten
    preds = torch.cat(preds, axis=0)
    labels = torch.cat(labels, axis=0)
    # Returns as numpy (CPU環境の場合は不要)
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return preds, labels


# 予測結果と正解値を取得
preds, labels = retrieve_result(model, test_loader)









# 学習済みモデルの保存・ロード
path_saved_model = "./saved_model"
# モデルの保存
torch.save(model.state_dict(), path_saved_model)
# モデルのロード
model = Mymodel()
model.load_state_dict(torch.load(path_saved_model))


# Model summary
from torchsummary import summary
model = model().to(device)
summary(model, input_size=(1, 50, 50))