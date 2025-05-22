import numpy as np

class Perceptron:
    # 생성자
    # thresholds : 임계값, 계산된 예측값을 비요하는
    # eta : 학습율
    # n_iner : 학습 횟수
    def __init__(self, thresholds=0.0, eta=0.01, n_iter=10):
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = np.zeros(1)
        self.errors_ = []

    # y = wx + b
    # 가중치 * 입력값 의 총 합 더하기 b를 구한다.
    def net_input(self, X):
        a1 = np.dot(X, self.w_[1:]) + self.w_[0]
        return a1

    # 학습 함수
    def fit(self, X, y):
        # 가중치를 담는 변수 X 의 컬럼수 + 1 해서 한자리는 b
        self.w_ = np.zeros(X.shape[1] + 1)
        # 예측 값, 실제 값 비교해서 다른 값이 나온 수
        self.errors_ = []

        for _ in range(self.n_iter):
            # 실제 값과 예측과 차이 난 개수
            errors = 0
            # 입력 받은 X, y를 하나로 묶는다.
            temp1 = zip(X, y)
            # 입력 값과 결과 값의 묶음을 가지고 반복.
            for xi, target in temp1:
                a1 = self.predict(xi)
                # 입력 값, 예측값이 다르면 가중치를 조정한다.
                if target != a1:
                    update = self.eta * (target - a1)
                    print("update: ", update)
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
                    print("w_: ", self.w_[1:], "b: ", self.w_[0])

            # 싥제 값과 예측 값이 다른 횟수 기록
            self.errors_.append(errors)

    # 예측 값 구하기.
    def predict(self, X):
        a2 = np.where(self.net_input(X) > self.thresholds, 1, -1)
        return a2
