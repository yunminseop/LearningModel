from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

        # padding이 무엇인가?
        # 실제 LLM 모델에 입력되는 문장의 단어 수는 천차만별인데, 이 문장의 길이가 다르다면 이들을 텐서로 묶어 배치로 처리하는 것이 불가능. (길이 5인 문장과 길이 10인 문장을 하나의 배치에 못 넣음)
        # 이를 해결하기 위해 배치 내의 문장들 중 가장 긴 문장의 길이로 통일하는 데 사용되는 것이 pad.
        # 나는 학교에 간다 [PAD]
        # 안녕하세요 [PAD] [PAD] [PAD]
        # 나는 집에 가고 싶다

        # 패딩 토큰의 임베딩은 항상 0으로 고정되어 역전파 과정에서 기울기 계산에 관여하지 않으며, attention score, weights, value 등을 구할 때 연산 과정을 오염시키지 않도록 Masking 처리 됨.