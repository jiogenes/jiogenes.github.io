---
layout: post
title: "세상에서 제일 쉬운 트랜스포머"
subtitle: 트랜스포머 구현과 학습 방법
categories: Transformer
comments: true
use_math: true
---

안녕하세요 jiogenes 입니다.

이번에는 transformer에 대해 공부해보겠습니다.

다른 블로그에서도 많이 다루고 있는 주제이긴 하지만 선수지식이 없는 상태에서 처음 트랜스포머를 배울때 코드가 정말 이해가 안됐던 기억이 있습니다.

제가 부족해서 그렇겠지만 저 말고도 다른 분들은 쉽게쉽게 이해하고 넘어가기 바라는 마음에서 코드구현상 어려운 부분을 모조리 제외하고 아주 쉽게 트랜스포머를 설명해 보고 직접 학습까지 돌려보도록 하겠습니다.

<img width="600" alt="Untitled" src="https://github.com/jiogenes/utterances/assets/43975730/385de395-a2fb-446a-addf-64f38de7b44e">

트랜스포머의 기본 설명은 제외하겠습니다. 이미 저 말고도 똑똑하신 분들이 설명을 잘해놓은 블로그들이 많아서 반복해서 설명하는것은 시간 낭비 일뿐 큰 도움이 되지 않을것 같네요.

트랜스포머 구조만 살펴보고 바로 코드로 들어가 보죠.

## 트랜스포머 구조

<img width="600" alt="Untitled 1" src="https://github.com/jiogenes/utterances/assets/43975730/bfbaa5b0-0143-48a9-b2df-d2ff5e773988">

익숙한 트랜스포머 구조입니다.

우선 구현해야 할 부분은 크게 3가지 입니다.

1. 학습 데이터 전처리
2. 트랜스포머 모델
3. 학습 코드 및 인퍼런스 코드

오늘 배울것은 트랜스포머 모델을 구현해보는 것이기 때문에 우선 트랜스포머 모델을 구현해 놓고 나머지 학습 데이터 처리와 트레이닝 및 인퍼런스 코드를 간략하게 작성해 보죠.

우리가 구현할 트랜스포머 모델의 클래스 다이어그램은 다음과 같습니다.


<img width="600" alt="Untitled 2" src="https://github.com/jiogenes/utterances/assets/43975730/e9dee881-94fb-4107-8563-dea8e70ab389">

구현은 작은 부분부터 구현해서 바텀-업 형식으로 구현하며 Huggingface의 transformers 라이브러리를 최대한 활용하여 구현해 보도록 하겠습니다.

## 토크나이저와 하이퍼파라미터

토크나이저와 각종 하이퍼파라미터는 기본 BERT 모델을 따라보겠습니다.

```python
import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
config = AutoConfig.from_pretrained('bert-base-uncased')
```

사실 다국어 언어모델을 사용하기 위해서는 다른 모델을 사용하는것이 일반적이지만 이번 시간에는 성능 보다는 구현에 초점을 맞춰보겠습니다. 트랜스포머의 성능을 높이는 방법에 대해서는 다음에 다뤄보도록 하겠습니다.

BERT의 하이퍼파라미터는 다음처럼 구성돼 있습니다.

```python
print(config)

>>>
BertConfig {
  "_name_or_path": "bert-base-uncased",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.35.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```

## 임베딩 레이어

임베딩 레이어는 자연어 인풋을 토큰 임베딩으로 바꿔주는 역할을 합니다. 그리고 트랜스포머는 셀프어텐션을 통해 인풋이 위치와 상관 없이 한번에 계산되므로 위치 임베딩을 더해줘야 합니다.

```python
class Embedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.positional_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout()

    def forward(self, x):
        positional_embedding = self.positional_embedding(torch.arange(x.size(1)).to(x.device))
        token_embedding = self.token_embedding(x)
        embedding = token_embedding + positional_embedding
        embedding = self.norm(embedding)
        embedding = self.dropout(embedding)
        return embedding
```

트랜스포머 논문에서는 sin, cos을 사용한 절대 위치 인코더를 사용하긴 하지만 위치 임베딩을 임베딩 레이어로 구현하는 경우도 많습니다. 문맥 내에서 유동적인 위치에 대해 학습이 가능하기 때문에 우리도 임베딩 레이어로 구현해 보겠습니다.

4번 라인에서 위치 임베딩을 선언하고 5번 라인에서 토큰 임베딩을 선언합니다. 위치 임베딩은 최대 입력 토큰 갯수만큼, 토큰 임베딩은 당연히 보캡 갯수만큼 선언해 줍니다. 위치 임베딩을 생성할 때 torch.arange로 만든 텐서의 디바이스도 맞춰줍니다.

12번 라인에서 토큰 임베딩과 위치 임베딩을 더해주면 임베딩 레이어의 역할은 끝입니다.

레이어놈과 드랍아웃은 효율적인 학습을 위해 추가합니다.

## 트랜스포머 인코더

### 셀프 어텐션

셀프 어텐션 레이어는 셀프 어텐션을 구하는 다음 수식을 그대로 구현하면 됩니다.

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, head_dim) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, head_dim)
        self.key = nn.Linear(hidden_dim, head_dim)
        self.value = nn.Linear(hidden_dim, head_dim)

    def forward(self, x, attention_mask=None):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        score = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(query.size(-1)))
        if attention_mask is not None:
            score = score.masked_fill(attention_mask.unsqueeze(1) == 0, -torch.inf)
        weight = F.softmax(score, dim=-1)
        return torch.bmm(weight, value)
```

쿼리, 키, 밸류 벡터 각각의 역할을 잘 이해하셨다면 구현 자체는 어렵지 않습니다.

임베딩된 입력 시퀀스 각각의 쿼리, 키, 밸류 벡터를 구하고난 후 14번 라인에서 쿼리와 키 벡터를 서로 내적해서 스코어를 구하고 스케일링, 마스킹, 소프트맥스를 차례대로 적용해 줍니다.

마스킹은 배치로 입력된 여러 문장들을 패딩 처리 할 때 패딩 토큰에 어탠션을 주지 않기 위해 사용됩니다. 그리고 입력 시퀀스 길이만큼 브로드케스팅 해주기 위해 두 번째 차원을 늘려줍니다.

소프트맥스를 가장 마지막 차원에 대해 수행해서 계산된 입력 시퀀스 각각의 스코어값들을 밸류 벡터에 곱할 가중치로 바꿔줍니다.

### 멀티 헤드 어텐션

멀티 헤드 어텐션은 셀프 어텐션을 여러개 만들면 됩니다.

```python
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_head = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_head

        self.self_attention = nn.ModuleList([SelfAttention(self.hidden_dim, self.head_dim) for _ in range(self.num_head)])
        self.output_weight = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x, attention_mask=None):
        x = torch.cat([a(x, attention_mask) for a in self.self_attention], dim=-1)
        return self.output_weight(x)
```

한 헤드에서 hidden dimension을 헤드 갯수로 나눈 길이만큼 출력하도록 셀프 어텐션을 만들어 줍니다. BERT의 hidden dimension은 768이고 헤드 갯수가 12개 이므로 각각 헤드의 출력은 64 입니다. 각각 헤드의 출력은 다시 concat 시켜서 원래 hidden dimension 길이로 맞춰줍니다. 사실 중간에 head dimension은 꼭 이 크기가 아니어도 상관없습니다. 마지막 output weight에서 최종 출력 길이를 조정할 수 있기 때문입니다.

그리고 8번 라인에서 Pytorch의 nn.Module 클래스에서 리스트 컴프리헨션 방식으로 만들어진 파라미터도 nn.ModuleList 를 통해 관리할 수 있도록 만들어 주는것도 잊어버리지 않도록 합니다.

### 피드 포워드

마지막으로 2개의 연결된 완전 연결 신경망을 통해 모델의 기억력을 증진합니다. 트랜스포머 구조를 가지는 대부분의 모델은 사이즈를 키울때 피드 포워드 층의 크기를 많이 늘립니다.

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
```

BERT의 intermediate size가 3072 이므로 hidden size의 4배로 확장했다가 hidden size로 최종 출력을 합니다. 그리고 트랜스포머 모델들에서 널리 사용되는 GELU 활성화 함수를 사용합니다.

### 인코더 레이어

위의 멀티 헤드 어텐션과 피드 포워드를 합쳐서 트랜스포머의 인코더 레이어 1개를 만들 수 있습니다.

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttentionLayer(config)
        self.feedforward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attention_mask=None):
        normed = self.norm1(x)
        x = self.attention(normed, attention_mask) + x
        normed = self.norm2(x)
        x = self.feedforward(normed) + x
        return x
```

그림에서 보듯이 멀티 헤드 어탠션과 피드 포워드의 출력에 layer norm과 skip connection을 사용하기 때문에 이것만 구현해 주면 됩니다. 따라서 2개의 layer norm을 선언해 줍니다.

트랜스포머 본문에서는 멀티 헤드 어텐션과 피드 포워드의 출력에 layer normalization을 수행하지만 이 후 많은 연구를 통해 입력에 layer norm을 미리 수행하면 학습이 더 안정적으로 된다는 사실이 밝혀졌습니다. 우리도 layer norm을 앞당겨 사전 층 정규화를 적용해 보죠.

그리고 멀티 헤드 어텐션과 피드 포워드 출력 각각에 입력값을 더해줍니다. 더해주는 입력값은 layer norm에 들어가기 전의 값을 넣어줍니다.

### 인코더

싱글 인코더 레이어를 여러개 쌓고 임베딩 레이어와 합치면 마침내 진정한 트랜스포머의 인코더가 됩니다.

```python
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        result = []
        for layer in self.encoder:
            x = layer(x, attention_mask)
            result.append(x)
        return result
```

각각의 인코더 레이어에서 나온 결과값을 디코더에서 사용하기위해 모두 저장해 리스트로 출력해 줍니다.

## 트랜스포머 디코더

### 마스크 셀프 어텐션

인코더와 디코더의 가장 큰 차이점이라면 마스크 셀프 어텐션의 사용여부입니다. 인코더는 입력 시퀀스를 다 볼 수 있기 때문에 이런 메커니즘이 필요 없지만 디코더는 입력 시퀀스를 다 볼 수 없기 때문에 마스크 메커니즘이 꼭 필요합니다.

```python
class MaskedSelfAttention(nn.Module):
    def __init__(self, hidden_dim, head_dim) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, head_dim)
        self.key = nn.Linear(hidden_dim, head_dim)
        self.value = nn.Linear(hidden_dim, head_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        score = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(query.size(-1)))
        mask = torch.tril(torch.ones((query.size(1), key.size(1)))).unsqueeze(0).to(x.device)
        score = score.masked_fill(mask == 0, -torch.inf)
        weight = F.softmax(score, dim=-1)
        return torch.bmm(weight, value)
```

14번 라인에서 어텐션 마스크가 아니라 torch.tril를 통해 삼각행렬 형태의 마스크를 만들어 줍니다. 입력된 타겟을 차례대로 보기 위함입니다.

### 마스크 멀티 헤드 어텐션

이 후 셀프 어텐션과 마찬가지로 마스크 셀프 어텐션을 여러 개 쌓아줍니다.

```python
class MaskedMultiHeadAttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_head = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_head

        self.self_attention = nn.ModuleList([MaskedSelfAttention(self.hidden_dim, self.head_dim) for _ in range(self.num_head)])
        self.output_weight = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        x = torch.cat([a(x) for a in self.self_attention], dim=-1)
        return self.output_weight(
```

### 크로스 어텐션

인코더의 셀프 어텐션 키, 벨류 값을 받아와서 디코더의 인풋 쿼리랑 계산하는 부분입니다. 이를 위해 각각의 쿼리, 키, 벨류 값을 따로따로 받도록 만들어 줍니다.

```python
class CrossSelfAttention(nn.Module):
    def __init__(self, hidden_dim, head_dim) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, head_dim)
        self.key = nn.Linear(hidden_dim, head_dim)
        self.value = nn.Linear(hidden_dim, head_dim)

    def forward(self, src, tgt, attention_mask=None):
        query = self.query(tgt)
        key = self.key(src)
        value = self.value(src)

        score = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(query.size(-1)))
        if attention_mask is not None:
            score = score.masked_fill(attention_mask.unsqueeze(1) == 0, -torch.inf)
        weight = F.softmax(score, dim=-1)
        return torch.bmm(weight, value)
```

쿼리는 타겟 시퀀스로 부터, 키와 벨류는 입력 시퀀스로 부터 계산되므로 입력과 타겟 시퀀스를 받아 쿼리, 키, 벨류를 계산해 줍니다.

어텐션 마스크는 입력 시퀀스의 어텐션 마스크를 통해 계산합니다.

### 크로스 멀티 헤드 어텐션

이 후 크로스 어텐션을 여러 층으로 쌓아 줍니다.

```python
class CrossMultiHeadAttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_head = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_head

        self.self_attention = nn.ModuleList([CrossSelfAttention(self.hidden_dim, self.head_dim) for _ in range(self.num_head)])
        self.output_weight = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, src, tgt, attention_mask=None):
        x = torch.cat([a(src, tgt, attention_mask) for a in self.self_attention], dim=-1)
        return self.output_weight(x)
```

### 디코더 레이어

위의 코드를 통해 싱글 디코더 레이어 하나를 만들 수 있습니다.

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.masked_attention = MaskedMultiHeadAttentionLayer(config)
        self.cross_attention = CrossMultiHeadAttentionLayer(config)
        self.feedforward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)

    def forward(self, src, tgt, attention_mask=None):
        normed = self.norm1(tgt)
        x = self.masked_attention(normed) + tgt
        normed = self.norm2(x)
        x = self.cross_attention(src, x, attention_mask) + x
        normed = self.norm3(x)
        x = self.feedforward(x) + x
        return x
```

그림에서 보듯이 인코더에 비해 마스크 멀티 헤드 어텐션과 크로스 어텐션이 추가되었고 인코더에서 오는 키, 벨류 값을 받을 수 있습니다.

### 디코더

싱글 디코더 레이어를 여러개 쌓고 임베딩 레이어를 추가하면 마침내 디코더를 완성할 수 있습니다!

```python
class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.decoder = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, src, tgt, attention_mask=None):
        x = self.embedding(tgt)
        for s, layer in zip(src, self.decoder):
            x = layer(s, x, attention_mask)
        logits = self.linear(x)
        return logits
```

10번 라인에서 인코더에서 넘어오는 hidden value를 각 레이어로 넘겨줍니다.

그리고 마지막으로 다음 단어를 예측하기 위해 linear레이어로 vocab size만큼 늘린 logit값을 출력해줍니다.

## 트랜스포머

```python
class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

    def forward(self, src, tgt, attention_mask=None):
        encoder_output = self.encoder(src, attention_mask)
        decoder_output = self.decoder(encoder_output, tgt, attention_mask)
        return decoder_output

    def inference(self, src):
        encoder_output = self.encoder(src)
        tgt = torch.tensor([[tokenizer.cls_token_id]])
        while tgt[0][-1] != tokenizer.sep_token_id and tgt.size(-1) < self.config.max_position_embeddings:
            decoder_output = self.decoder(encoder_output, tgt)
            decoder_output = torch.argmax(decoder_output, dim=-1, keepdim=True)
            tgt = torch.concat([tgt, decoder_output[:, -1]], dim=-1)
        return tgt
```

최종 트랜스포머 코드는 인코더-디코더 구조로 짜여진 매우 간단한 형태의 코드가 됩니다. 가장 밑의 구조부터 힘들게 만들어 오니 최종적으로는 굉장히 추상화가 잘 되었습니다.

12번 라인의 inference 메소드는 트랜스포머를 다 학습하고 난 뒤 추론을 위한 코드입니다. 추론할 때는 타겟 시퀀스를 한꺼번에 넣어 줄 수 없으므로 반복문을 통해 이전에 생성된 결과물을 하나하나씩 붙여가는 형태로 만들어 줍니다. 처음에는 시작 토큰만 넣어 결과 값을 뽑고 이 후 나오는 결과 값의 마지막 값을 하나씩 뒤에다 concat 시켜 줍니다. eos 토큰이나 (BERT에서는 SEP 토큰) 최대 길이에 이르면 그만 출력하도록 하면 됩니다.

완성된 코드를 통해 학습을 진행해 봅시다.

## 데이터

데이터는 한글과 영어로 구성된 ted 강의 데이터를 사용해보겠습니다. [데이터 링크](https://huggingface.co/datasets/msarmi9/korean-english-multitarget-ted-talks-task)

huggingface 의 datasets 라이브러리를 통해 진행하므로 datasets 패키지를 설치해 주세요. [링크](https://huggingface.co/docs/datasets/installation)

데이터셋의 예시를 보면 ted강의의 한글본과 영어본이 한문장씩 나뉘어져 있습니다.

```python
from datasets import load_dataset

dataset = load_dataset('msarmi9/korean-english-multitarget-ted-talks-task')

print(dataset['train']['korean'][0])
print(dataset['train']['english'][0])

>>>
(박수) 이쪽은 Bill Lange 이고, 저는 David Gallo입니다
(Applause) David Gallo: This is Bill Lange. I'm Dave Gallo.
```

데이터가 몇개인지 살펴봅시다.

```python
print(dataset)

>>>
DatasetDict({
    train: Dataset({
        features: ['korean', 'english'],
        num_rows: 166215
    })
    validation: Dataset({
        features: ['korean', 'english'],
        num_rows: 1958
    })
    test: Dataset({
        features: ['korean', 'english'],
        num_rows: 1982
    })
})
```

데이터 양도 많고 훈련, 검증 세트가 적절히 나뉘어져 있어서 한국어 번역을 위한 데이터로써 정말 손색이 없습니다.

### 데이터 전처리

저는 개인적으로 트랜스포머 아키텍쳐를 학습하기 위해 데이터를 잘 뜯어보고 데이터와 친해지는 것이 더 중요하고 급선무라고 생각합니다. 하지만 다시 한 번 이번 블로그 내용은 트랜스포머 아키텍쳐를 어떻게 만들고 학습하는가에 대해서만 다루는 것임을 짚고 넘어가보겠습니다 😅 

우선 datasets 라이브러리의 사용이유인 map함수를 사용해서 전처리를 진행해 봅시다.

```python
def preprocess_func(examples):
    inputs = [e for e in examples['english']]
    targets = [e for e in examples['korean']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=config.max_position_embeddings, truncation=True)
    return model_inputs

tokenized_dataset = dataset.map(preprocess_func, batched=True, remove_columns=dataset['train'].column_names)
print(tokenized_dataset)

>>>
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 166215
    })
    validation: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 1958
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 1982
    })
})
```

위 코드를 통해 인풋 시퀀스를 영어로, 타겟 시퀀스를 한글로 바꾼 후 BERT tokenizer를 통해 인코딩을 수행합니다.  그리고 원래 있던 english와 korean 컬럼을 삭제하고 모델의 인풋에 맞게 input_ids, attention_mask, labels로 변경합니다. token_type_ids는 BERT 모델에서 NSP 훈련을 위한 컬럼이니 신경쓰지 않아도 됩니다.

그리고 효율적인 학습을 위해 데이터를 배치 형태로 만들어 봅시다.

```python
def collate_fn(batch):
    input_ids = [torch.tensor(example['input_ids']) for example in batch]
    attention_mask = [torch.tensor(example['attention_mask']) for example in batch]
    labels = [torch.tensor(example['labels']) for example in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels}

train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(tokenized_dataset['validation'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=1, shuffle=False, collate_fn=collate_fn)
```

pytorch의 데이터로더를 이용해서 배치형태의 데이터를 만듭니다. collate_fn 메소드를 직접 정의해서 앞서 map 함수를 통해 만든 데이터를 torch의 텐서로 바꿔주고 pad_sequence 함수를 통해 배치 형태 내부의 서로다른 시퀀스 길이를 패딩 토큰을 추가하여 통일해줍니다.

## 학습 및 추론

### 학습

학습 코드는 다른 모델들과 큰 차이가 없습니다. 다만 주의해야 할 점들이 있습니다.

1. pad_sequnce함수에서 추가된 패딩 토큰은 학습되지 않도록 만들어야 합니다. 따라서 다음 토큰을 맞추기 위해 계산되는 loss 함수에서 패딩 토큰은 무시하도록 해줍니다
2. loss를 계산할 때 다음 토큰을 맞춰야 하므로 모델의 입력에는 왼쪽으로 한칸 씩 밀린 값을 넣고 정답은 오른쪽으로 한칸 씩 밀린 값을 맞추도록 합니다.

어떻게 구현하는지 살펴보죠.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
learning_rate = 5e-5
epochs = 1

model = Transformer(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

for e in range(epochs):
    model.train()
    train_loss = .0
    for batch in tqdm(train_dataloader, ncols=80, ascii=True, desc='train_step'):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(src=batch['input_ids'], tgt=batch['labels'][:, :-1], attention_mask=batch['attention_mask'])
        loss = loss_func(outputs.transpose(1, 2), batch['labels'][:, 1:])

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(batch['input_ids'])

    train_loss /= len(train_dataloader.dataset)
    print(train_loss)

    model.eval()
    test_loss = .0
    for batch in tqdm(valid_dataloader, ncols=80, ascii=True, desc='test_step'):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(src=batch['input_ids'], tgt=batch['labels'][:, :-1], attention_mask=batch['attention_mask'])
        loss = loss_func(outputs.transpose(1, 2), batch['labels'][:, 1:])

        loss.backward()

        test_loss += loss.item() * len(batch['input_ids'])

    test_loss /= len(valid_dataloader.dataset)
    print(test_loss)

    torch.save(model.state_dict(), f'./weight_{e+1}.pt')
```

8번 라인에서 ignore_index 파라미터를 통해 패딩 토큰은 loss를 계산하는데서 무시되도록 설정합니다.

18, 19번 라인에서 batch[’labels’]의 값을 맨끝에서 한 개 자르고 맨 처음에서 한 개 잘라서 학습하도록 만듭니다.

저는 이 코드를 1080Ti에서 돌렸습니다. 메모리의 한계로 배치사이즈도 4가 최대고 시간도 한 에폭당 약 10시간이 걸려 1에폭만 학습하도록 진행했습니다.

### 추론

추론은 간단하게 실행해보도록 하겠습니다. 토크나이저가 다중언어 모델에 맞지 않고 학습도 너무 적게 돌려서 제대로 작동하지 않네요…. 😭 

```python
transformer = Transformer(config)
transformer.load_state_dict(torch.load('./weight_1.pt'))
sample = tokenizer.encode("(Applause) David Gallo: This is Bill Lange. I'm Dave Gallo.", return_tensors='pt')
result = transformer.inference(sample)
tokenizer.decode(result[0])

>>>
[CLS] ( 박수 ) 이런 일이 [UNK]. 이런 [UNK]. [SEP]
```

<img width="611" alt="Untitled 3" src="https://github.com/jiogenes/utterances/assets/43975730/da9c5530-7996-49f0-aa8a-3c07e827c82e">

우리가 구현한 트랜스포머는 확실히 옛날 구식 모델이지만 최근의 LLM들 모두 이 트랜스포머 구조를 크게 벗어난 모델이 없습니다. 트랜스포머를 제대로 이해하고 구현과 학습을 잘 할줄 안다면 자연어 처리 분야에서 최고의 무기를 가진것이나 다름 없습니다.

구현한 코드를 좀 더 빠르게 학습하는 방법, 학습 효율을 높이는 방법 등 트랜스포머를 좀 더 활용하는 방법은 추후에 작성하도록 하겠습니다.