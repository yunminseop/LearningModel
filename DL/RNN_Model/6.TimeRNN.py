import numpy as np
class RNN:
    def _init_(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)
        self.cache = (x, h_prev, h_next)
        return h_next
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache
        dt = dh_next*(1-h_next**2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        print(f"Wx: {self.grads[0]}")
        print(f"Wx.shape: {self.grads[0].shape}")
        print(f"Wh: {self.grads[1]}")
        print(f"Wh.shape: {self.grads[1].shape}")
        print(f"b: {self.grads[2]}")
        print(f"b.shape: {self.grads[2].shape}")
        return dx, dh_prev
class TimeRNN:
    def _init_(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads =[np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful
    def set_state(self, h):
        self.h = h
    def reset_state(self):
        self.h = None
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape
        self.layers = []
        hs = np.empty((N, T, H), dtype='f') # N = 4, T = 5, D = 3, H = 6
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
            for t in range(T):
                layer = RNN(*self.params)
                print(layer.grads)
                self.h = layer.forward(xs[:, t, :], self.h) # 각 타임스텝마다 순회하면서 순전파 진행. 주의: layer 변수는 TimeRNN 인스턴스가 아니라 RNN 인스턴스임!
                hs[:, t, :] = self.h
                # x[:, t, :], h[:, t, :] -> 각 타임스텝에 해당하는 값을 반복문을 돌며 업데이트
                self.layers.append(layer) # 각 레이어에 대해 RNN layer을 저장
            print("-----------------")
        return hs
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape
        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            print(t)
            print(f"dh: {dh}")
            dx, dh = layer.backward(dhs[:, t, :] + dh) # layer 객체는 RNN 인스턴스임. backward 메서드 사용하여 TimeRNN 마지막 레이어에서부터 순회하며 Backpropagation
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                print(grad)
                grads[i] += grad
            print(f"grads[{i}]: {grads[i]}")
            print(f"grads[{i}]: {grads[i].shape}")
            print("-----------------")
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs
if _name_ == "_main_":
    N, T, D, H = 4, 5, 3, 6  # 배치 4개, 시퀀스 길이 5, 입력 차원 3, hidden 차원 6
    np.random.seed(42)
    xs = np.random.randn(N, T, D).astype('f')
    dhs = np.random.randn(N, T, H).astype('f')
    # RNN 가중치 초기화
    Wx = np.random.randn(D, H).astype('f') * 0.01
    Wh = np.random.randn(H, H).astype('f') * 0.01
    b = np.zeros(H, dtype='f')
    # TimeRNN 생성 및 실행
    rnn = TimeRNN(Wx, Wh, b)
    hs = rnn.forward(xs)
    dxs = rnn.backward(dhs)
    # 출력 확인
    print(">>> Hidden States (hs):", hs.shape)
    print(">>> dxs:", dxs.shape)
    print(">>> dh (from last timestep):", rnn.dh.shape)
    print(">>> Wx Gradient norm:", np.linalg.norm(rnn.grads[0]))
    print(">>> Wh Gradient norm:", np.linalg.norm(rnn.grads[1]))
    print(">>> b Gradient norm:", np.linalg.norm(rnn.grads[2]))