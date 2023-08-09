---
layout: post
title: "[Effective Python] 리스트 컴프리헨션"
subtitle: map과 filter를 대신하는 list comprehension
categories: Python
comments: true
use_math: true
---

| 이 글은 [Effective Python](https://effectivepython.com/)을 참고하여 유용하다고 생각하는 방법을 선정하여 작성한 글입니다.

안녕하세요. jiogenes 입니다. 오늘은 리스트 컴프리헨션(list comprehension; 리스트 함축 표현식)에 대해 알아보겠습니다.

파이썬에는 한 리스트를 통해 다른 리스트를 만드는 리스트 컴프리헨션이라는 간단한 문법이 있습니다.

### 기본 문법

간단한 예시로 리스트에 있는 각 숫자의 제곱을 계산하고 싶다면 다음과 같이 만들 수 있습니다.

```python
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares = [x ** 2 for x in a]
print(a)

>>>
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

또 짝수만 걸러내고 싶다면 다음과 같이 만들 수 있습니다.

```python
even_squares = [x ** 2 for x in a if x % 2 == 0]
print(even_squares)

>>>
[4, 16, 36, 64, 100]
```

이러한 연산은 python 내장함수인 `map`과 `filter`로도 가능합니다.

```python
alt_squares = map(lambda x: x ** 2, a)
alt_even_squares = map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, a))
assert squares == list(alt_squares)
assert even_squares = list(alt_even_squares)
```

하지만 `map`과 `filter`함수를 사용하면 내부에 `lamda`식이 들어가야 하기 때문에 깔끔하지 않고 출력 결과를 `list()`로 감싸야 리스트로 사용할 수 있습니다. **따라서 `map`과 `filter` 대신 리스트 컴프리헨션을 사용하는 것이 더 파이썬스러운 코딩 방법 이라고 할 수 있습니다.**

### 딕셔너리와 세트 컴프리헨션

딕셔너리와 세트에도 컴프리헨션 문법을 사용할 수 있습니다. 컴프리헨션 문법을 사용하면 직관적으로 파생되는 자료구조를 쉽게 설명할 수 있습니다. 딕셔너리는 중괄호 `{}`안에 key, value를 나누는 세미콜론 `:`이 들어가야 하고 세트는 리스트 컴프리헨션에서 대괄호 `[]`를 중괄호 `{}`로 바꿔주면 됩니다.

```python
token_to_id = {'apple': 1, 'banana': 2, 'cherry': 3}
id_to_token = {id: token for token, id in token_to_id.items()}
token_len_set = {len(token) for token in token_to_id.keys()}
print(id_to_token)
print(token_len_set)

>>>
{1: 'apple', 2: 'banana', 3: 'cherry'}
{5, 6}
```

### 튜플과 제네레이터

여기까지 봤다면 튜플도 똑같이 소괄호 `()`를 통해 컴프리헨션 문법으로 만들 수 있다고 생각할 수 있겠습니다. 하지만 튜플은 컴프리헨션으로 만들려면 `tuple()`이 명시적으로 작성되어야 합니다. 명시적으로 작성되지 않고 소괄호 `()`만 사용할 경우 제네레이터가 생성됩니다.

```python
is_tuple = (i for i in range(10))
print(is_tuple)
is_tuple = tuple(i for i in range(10))
print(is_tuple)

>>>
<generator object <genexpr> at 0x104ab9b30>
(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
```

위와같이 튜플을 컴프리헨션으로 사용하기 위해서는 앞에 `tuple()`을 꼭 써줘야 합니다.

제네레이터는 엄청나게 긴 시퀀스를 다룰 때 한번에 메모리에 올리기 힘든 경우 사용하는 함수 또는 반복자입니다. 제네레이터는 다음에 다루도록 하겠습니다.

### 리스트 컴프리헨션 내부 표현식

파이썬의 리스트 컴프리헨션은 다중 루프도 지원합니다. 예를들어 2차원 행렬을 1차원으로 만들 때는 다음과 같이 사용할 수 있습니다.

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [x for row in matrix for x in row]
print(flat)

>>>
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

그리고 다음과 같이 리스트 컴프리헨션 내부에 또 다른 리스트 컴프리헨션이 들어가도록 구성할 수도 있습니다.

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
squared = [[x ** 2 for x in row] for row in matrix]
print(squared)

>>>
[[1, 4, 9], [16, 25, 36], [49, 64, 81]]
```

이러한 문법들은 내부에 표현식이 2개 중복되어 있어 크게 좋아보이진 않지만 그래도 이해할 수는 있습니다. 하지만 3차원 행렬을 다룰때 리스트 컴프리헨션을 사용한다면 리스트 컴프리헨션을 여러줄로 구분해야 할 정도로 길어집니다.

```python
matrix = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
]
flat = [x for sublist1 in matrix
        for sublist2 in sublist1
        for x in sublist2]
print(flat)

>>>
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

이러한 문법은 짧지도 않고 이해하기 쉽지 않습니다. 오히려 일반 루프문을 썼을때 더 보기좋고 이해하기 쉽습니다.

```python
flat = []
for sublist1 in matrix:
    for sublist2 in sublist1:
        flat.extend(sublist2d)
```

리스트 컴프리헨션은 다중 if문도 지원합니다. 그리고 같은 레벨에 여러 조건이 있으면 암시적인 and 표현식이 됩니다. 예를들어 4이상의 짝수인 리스트를 생성하는 방법은 다음과 같습니다.

```python
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
method1 = [x for x in a if x >= 4 if x % 2 == 0]
method2 = [x for x in a if x >= 4 and x % 2 == 0]

assert method1 == method2
```

다중 루프와 조건문이 섞인 리스트 컴프리헨션도 작성할 수 있습니다. 행렬의 행의 합이 10 이상이고 3으로 나누어 떨어지는 원소를 구하고 싶다면 다음과 같이 코드를 작성할 수 있습니다.

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
filtered = [[x for x in row if x % 3 == 0] 
            for row in matrix if sum(row) >= 10]
print(filtered)

>>>
[[6], [9]]
```

리스트 컴프리헨션을 활용하다 보면 이렇게 다중 루프와 다중 조건식을 마구마구 사용하고 싶을때가 많습니다. 그리고 이러한 표현방법이 무엇인가 파이썬스럽다는 느낌을 받을때가 있습니다. 하지만 파이썬은 쉽고 간결하게 작성하는 것은 맞지만 보는사람이 이해하기 쉬워야 합니다. **몇 줄 절약한 장점이 나중에 겪을 어려움보다 크지 않습니다.**

저자는 리스트 컴프리헨션을 사용할 때 표현식이 2개를 넘어가는것을 피하라고 권장합니다. 조건문 2개, 루프 2개 혹은 조건 1개 및 루프 1개 정도가 이해하기 적당하다고 합니다. 이것보다 복잡한 상황에서는 일반적인 if문과 for문을 활용하는 것을 추천합니다.

### 정리

- 내장 함수인 map이나 filter를 사용하는 것보다 직관적인 리스트 컴프리헨션을 사용하자.
- 딕셔너리와 세트도 컴프리헨션 표현식을 지원한다.
- 컴프리헨션 표현식은 다중 루프와 다중 조건을 지원한다.
- 컴프리헨션의 내부 표현식이 두 개가 넘어가는 것을 지양하자.
