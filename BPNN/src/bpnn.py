"""
BP神经网络模型
四层：输入层-embedding层-隐含层(一层)-输出层
"""
from utils import *
from config import *
from collections import defaultdict

np.random.seed(random_seed)


class BPNN:
    def __init__(self, embedding, output_layer_size):
        self.activation = activation  # 隐藏层的激活函数
        self.Lambda = Lambda  # L2正则化项系数λ
        self.learning_rate = learning_rate  # 学习率
        self.shuffle = shuffle  # 在划分batch的时候是否打乱数据集
        self.embedding_trainable = embedding_trainable  # 是否训练词向量
        self.weight_decay = 1  # 权重衰减

        self.embedding_matrix = embedding.embedding_matrix  # embedding对象
        self.embedding_dim = embedding.embedding_dim
        self.input_layer_size = embedding.window * embedding.embedding_dim  # 输入层中的神经元数量
        self.hidden_layer_size = hidden_layer_size  # 隐藏层中的神经元数量
        self.output_layer_size = output_layer_size  # 输出层中的神经元数量
        self.layer_sizes = [self.input_layer_size, self.hidden_layer_size, self.output_layer_size]  # 每层神经元数目
        self.layer_num = len(self.layer_sizes)
        self.all_w = [np.random.randn(l2, l1) / np.sqrt(l1) for l1, l2 in
                      zip(self.layer_sizes[:-1], self.layer_sizes[1:])]  # 权重,矩阵维度(下层特征数量*上层特征数量)
        self.all_b = [np.random.randn(l, 1) for l in self.layer_sizes[1:]]  # 偏置

    def forward(self, x):
        """
        前向传播
        :param x: 在embedding_matrix中,每行是一个词向量,x是当前词在embedding_matrix的行索引列表
        :return:
        all_z：保存所有层神经元的第一层输出,即x用权重和偏置计算后的输出z = wx + b
        all_a：保存所有层神经元的第二层输出,即z进激活函数后的输出a=f(z)
        """
        a = np.reshape(self.embedding_matrix[x], (-1, 1))  # window个词的行向量列表转为一列向量
        all_z = []
        all_a = [a]  # embedding层的输出是a
        for W, b in zip(self.all_w[:-1], self.all_b[:-1]):
            z = W @ a + b  # 本层所有神经元的第一层输出,即x用权重和偏置计算后的输出z = wa + b
            all_z.append(z)
            if self.activation == 'relu':  # 选择relu作为本层激活函数
                a = relu(z)
            elif self.activation == 'sigmoid':  # 选择sigmoid作为本层激活函数
                a = sigmoid(z)
            all_a.append(a)
        z = self.all_w[-1] @ a + self.all_b[-1]  # 本层所有神经元的第二层输出,即z进激活函数后的输出a=f(z)
        all_z.append(z)
        a = softmax(z)  # softmax作为输出层激活函数,将输出转变成概率密度函数
        all_a.append(a)
        return all_z, all_a

    def backward(self, all_z, all_a, y):
        """
        反向传播
        :param all_z: 保存所有层神经元的第一层输出,即x用权重和偏置计算后的输出z = wx + b
        :param all_a: 保存所有层神经元的第二层输出,即z进激活函数后的输出a=f(z)
        :param y: ont-hot形式的标签
        :return:
        dW：权重的梯度
        db：偏置的梯度
        dX：embedding层的梯度
        """
        dw = [np.zeros(w.shape) for w in self.all_w]  # 所有权重梯度
        db = [np.zeros(b.shape) for b in self.all_b]  # 所有偏置梯度
        dz = all_a[-1] - y  # 交叉熵损失函数对z求导, dL / dz = (dL / da) * (da / dz) = pi - yi
        db[-1] = dz  # z = wx + b, dL / db = dL / dz
        dw[-1] = dz @ all_a[-2].T  # z = wa + b, dL / dw = (dL / dz) * a, a是输出层输入即倒数第二层输出
        for i in range(2, self.layer_num):  # 根据链式法则从后向前反向传播
            z = all_z[-i]  # z = wa + b, a是本层输入即前层输出
            da = self.all_w[-i + 1].T @ dz  # dL / da = (dL / dz) * w
            if self.activation == 'relu':  # a=f(z),f()是激活函数
                dz = da * relu_derivative(z)  # dL / dz = (dL / da) * (df / dz)
            elif self.activation == 'sigmoid':
                dz = da * sigmoid_derivative(z)
            dw[-i] = dz @ all_a[-i - 1].T  # z = wa + b, dL / dw = (dL / dz) * a
            db[-i] = dz  # z = wx + b, dL / db = dL / dz
        dx = self.all_w[0].T @ dz  # z = wx + b, dL / dx = (dL / dz) * w
        return dw, db, dx

    def gradient_descent(self, sum_dw, sum_db, sum_dx, m):  # 梯度下降
        """
        梯度下降法更新权重、偏置和embedding矩阵
        :param sum_dw:权重梯度
        :param sum_db:偏置梯度
        :param sum_dx:embedding矩阵梯度
        :param m:embedding:batch size
        :return:
        """
        self.all_w = [self.weight_decay * w - self.learning_rate * sum_grad / m
                      for w, sum_grad in zip(self.all_w, sum_dw)]
        self.all_b = [b - (self.learning_rate * sum_grad / m) for b, sum_grad in zip(self.all_b, sum_db)]
        if self.embedding_trainable:
            for word_id, sum_grad in sum_dx.items():
                self.embedding_matrix[word_id] -= (self.learning_rate * sum_grad / m)

    def loss(self, data):
        """
        loss function(加入正则项的交叉熵损失函数)
        C = (-1/n)*[∑i yi * log(pi)] + [λ/(2*n) * ω^2]  = (1/n) * (-∑i yi * log(pi) + [λ/2 * ω^2])
        :param data: 数据
        :return:
        """
        loss = 0.0
        for x, y in data:
            _, all_a = self.forward(x)  # 前向传播获得所有层输出
            a = all_a[-1]  # 取输出层输出
            loss += -np.log(a[np.argmax(y)])  # 交叉熵, -∑i yi * log(pi) , yi=[0...1...0] ,只加yi=1对应的log(pi)即可
        loss += 0.5 * self.Lambda * sum([np.linalg.norm(w) for w in self.all_w])  # 加入正则项, λ/2 * ω^2
        loss /= len(data)
        return loss

    def evaluate(self, data):
        """
        在数据集上评价模型
        :param data:用于评价的数据
        :return:
        correct_num:正确样本数量
        all_num:总样本数量
        accuracy:正确率
        """
        all_num = len(data)
        predict = []  # 预测的标签
        for x, y in data:
            all_z, all_a = self.forward(x)
            a = all_a[-1]
            predict.append(y[np.argmax(a)][0])  # 只有预测正确了y[np.argmax(a)]才等于1否则等于0
        correct_num = int(sum(predict))
        accuracy = correct_num / all_num
        return correct_num, all_num, accuracy

    def fit(self, data):  # 训练模型
        m = len(data)
        sum_dw = [np.zeros(w.shape) for w in self.all_w]  # 权重梯度的和
        sum_db = [np.zeros(b.shape) for b in self.all_b]  # 偏置梯度的和
        sum_dx = defaultdict(float)  # 输入矩阵太稀疏,采用字典更新,total_word_num * window 矩阵 -> unique_word_num 字典
        for x, y in data:
            all_z, all_a = self.forward(x)
            dw, db, dx = self.backward(all_z, all_a, y)
            sum_dw = [sum_dw[i] + dw[i] for i in range(len(dw))]
            sum_db = [sum_db[i] + db[i] for i in range(len(db))]
            for word_id, grad in zip(x, np.reshape(dx, (-1, self.embedding_dim))):  # 分离成window个词向量梯度
                sum_dx[word_id] += grad
        self.gradient_descent(sum_dw, sum_db, sum_dx, m)  # 根据求出的梯度更新W和b,以及embedding
