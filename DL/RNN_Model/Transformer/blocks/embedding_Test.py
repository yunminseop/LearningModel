import torch
import torch.nn as nn

vocab_size = 10000  # 단어 사전의 크기
embedding_dim = 128 # 임베딩 벡터의 차원
padding_token_idx = 0 # 패딩 토큰으로 사용할 인덱스 (예: 0번)

# Embedding 레이어 생성
embedding_layer = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=padding_token_idx
)

# 입력 시퀀스 예시
# 0은 패딩 토큰, 1, 2, 3은 다른 단어 토큰
input_sequence = torch.tensor([[1, 2, 3, 0, 0],
                               [4, 5, 0, 0, 0]])

# 임베딩 수행
embedded_sequence = embedding_layer(input_sequence)

print("임베딩 결과의 shape:", embedded_sequence.shape)
print("패딩 토큰 (0번 인덱스)의 임베딩:", embedding_layer.weight[padding_token_idx])
# 출력: 패딩 토큰의 임베딩은 항상 0 벡터임을 확인할 수 있습니다.