---
layout: post
title: "[Effective Python] 제네레이터"
subtitle: 제네레이터 사용법과 장점 및 사용시 주의점
categories: Python
comments: true
use_math: true
---

| 이 글은 [Effective Python](https://effectivepython.com/)을 참고하여 유용하다고 생각하는 방법을 선정하여 작성한 글입니다.

안녕하세요. jiogenes 입니다. 오늘은 파이썬의 제네레이터(generator)에 대해 알아보겠습니다.

파이썬 제네레이터는 리스트나 튜플과 같은 시퀀스와 달리 한번에 모든 원소들을 메모리에 올리지 않습니다.

예를들어 다음과 같이 리스트 컴프리헨션으로 파일의 모든 라인을 읽어들이는 코드는 파일의 크기에 따라 메모리를 소모하기 때문에 파일 크기가 작을 때는 괜찮지만 파일의 크기가 클 때는 좋지 않은 코드입니다.

```python
value = [len(x) for x in open('./my_file.txt')]
print(value)

>>>
[100, 57, 15, 1, 12, 75, 5, 86, 89, 11]
```

### 기본 문법

파이썬은 이러한 문제를 해결하기 위해 제네레이터 표현식(generator expression)을 제공합니다. 제네레이터 표현식은 `()`를 사용하고 리스트 컴프리헨션과 같은 문법을 사용합니다. 표현식을 사용하면 시퀀스를 한 번에 로딩하지 않고 한번에 한 아이템씩 출력하는 이터레이터(iterator)로 평가됩니다.

```python
it = (len(x) for x in open('./my_file.txt'))
print(it)

>>>
<generator object <genexpr> at 0x105f55d60>
```

제네레이터를 생성한 뒤 다음 출력값을 사용하고 싶다면 내장함수 `next`를 사용합니다.

```python
print(next(it))
print(next(it))

>>>
100
57
```

제네레이터를 만드는 또 다른 방법으로는 `yield`를 사용하는 것입니다.

```python
def my_func():
    for x in open('./my_file.txt'):
        yield x

it = my_func()
print(it)

>>>
<generator object it at 0x10a953a50>
```

제네레이터 표현식과 마찬가지로 사용방법은 같습니다.

```python
print(next(it))
print(next(it))

>>>
100
57
```

### 제네레이터로 함수 속 리스트 제거

제네레이터는 메모리를 아끼는 역할도 하지만 코드를 좀 더 깔끔하고 이해하기 쉽게 만들어 주기도 합니다.

예를들어 입력된 문장속 단어들의 첫번째 인덱스를 반환하는 함수를 작성하고 싶다면 리스트를 통해 다음과 같이 만들 수 있습니다.

```python
def index_words(text):
    result = []
    if text:
        result.append(0)
    for index, letter in enumerate(text):
        if letter == ' ':
            result.append(index + 1)
    return result

address = 'Four score and seven years ago...'
result = index_words(address)
print(result[:3])

>>>
[0, 5, 11]
```

위 함수는 리스트를 사용하기 때문에 리스트를 선언하는 라인과 `append` 하는 라인 그리고 리스트를 반환하는 라인이 필요하기 때문에 코드가 길어지고 그로인해 한번에 이해하기 쉽지 않습니다. 또한 `append` 때문에 `index + 1` 이라는 구문이 덜 부각됩니다. 제네레이터를 사용한다면 다음과 같이 작성할 수 있습니다.

```python
def index_words_iter(text):
    if text:
        yield 0
    for index, letter in enumerate(text):
        if letter == ' ':
            yield index + 1

result = list(index_words_iter(address))
```

함수 내 리스트 선언 부분과 리스트와 연동하는 부분들이 사라져 코드가 매우 간단해지고 이해하기 쉬워졌습니다. 그리고 내장함수 `list`로 감싸주면 손쉽게 리스트로 변환할 수 있습니다.

### 제네레이터 중첩

제네레이터 표현식은 다른 제네레이터 표현식과 중첩해서 사용할 수 있습니다. 여러 제네레이터를 중첩해서 사용하면 리스트를 중첩해서 사용하는 것 보다 획기적으로 메모리를 줄일 수 있습니다. 하지만 리스트 중첩과 달리 중첩된 모든 제네레이터가 한 번에 한 번씩 아이템을 꺼내므로 주의해서 사용해야 합니다.

```python
roots = ((x, x ** 0.5) for x in it)
print(next(roots))

>>>
(15, 3.872983346207417)
```

### 제네레이터 인수 순회

시퀀스를 참조할 때 시퀀스를 여러번 순회하는 경우가 있습니다. 이런 경우 리스트나 튜플의 경우에는 별 문제가 되지 않지만 그 객체가 제네레이터일 때는 주의가 필요합니다.

예를들어 다음과 같이 입력받은 리스트의 값을 정규화 하는 함수를 작성할 때 리스트를 여러번 순회할 수 있습니다.

```python
def normalize(inputs):
    total = sum(inputs)
    result = []
    for value in inputs:
        percent = 100 * value / total
        result.append(percent)
    return result

inputs = [15, 35, 80]
percentages = normalize(inputs)
print(percentages)

>>>
[11.538461538461538, 26.923076923076923, 61.53846153846154]
```

입력할 리스트가 파일에 저장되어 있다면 파일에서 한 줄 씩 읽어들이는 제네레이터를 사용할 수 있습니다.

```python
def read_file(data_path):
    with open(data_path) as f:
        for line in f:
            yield int(line)
```

`read_file` 함수를 사용하여 만든 제네레이터를 `normalize` 함수에 집어넣으면 아무 결과도 생성되지 않습니다.

```python
it = read_file('./my_file.txt')
percentages = normalize(it)
print(percentages)

>>>
[]
```

이것은 이터레이터(iterator)가 결과를 딱 한번만 생성하기 때문에 일어나는 상황입니다. 이미 `StopIteration` 예외가 발생한 이터레이터나 제네레이터는 또 호출하더라도 아무 결과도 얻을 수 없습니다.

```python
it = read_file('./my_file.txt')
print(list(it)) # 여기서 한번만 생성
print(list(it)) 

>>>
[15, 35, 80]
[]
```

이러한 문제를 해결하기 위한 가장 간단한 방법은 입력된 이터레이터를 명시적으로 복사해서 사용하는 것입니다. 하지만 이는 입력 이터레이터가 클 때 문제가 될 뿐더러 우리가 제네레이터를 사용한 목적과 애초에 부합하지 않습니다.

다른 해결방법으로는 새로운 이터레이터를 반환하는 함수를 받도록 만드는 것입니다.

```python
def normalize_func(get_iter):
    total = sum(get_iter())
    result = []
    for value in get_iter():
        percent = 100 * value / total
        result.append(percent)
    return result

percentages = normalize_func(lambda: read_fule(path))
print(percentages)

>>>
[11.538461538461538, 26.923076923076923, 61.53846153846154]
```

원하는 대로 동작하긴 하지만 람다 함수를 사용하는 것은 복잡하고 깔끔해 보이지 않습니다. 이것보다 훨씬 더 좋은 방법은 이터레이터 프로토콜(iterator protocol)을 구현한 새 컨테이터 클래스를 구현하는 것입니다. 클래스를 선언하고 내부에 매직메소드인 `__iter__`를 구현하면 이터레이터 프로토콜을 따르는 컨테이너 클래스를 만들 수 있습니다.

```python
class ReadFile(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        with open(self.data_path) as f:
            for line in f:
                yield int(line)

it = ReadFile('./my_file.txt')
percentages = normalize(it)
print(percentages)

>>>
[11.538461538461538, 26.923076923076923, 61.53846153846154]
```

이렇게 이터레이터 프로토콜을 따르는 컨테이너 클래스를 구현하면 이터레이터를 순회하고자 할 때 항상 매직 메소드 `__iter__`를 호출하므로 새로운 이터레이터를 할당받기 때문에 같은 컨테이너를 여러번 순회하더라도 정상적으로 동작하게 됩니다.

마지막으로 일반 제네레이터 혹은 이터레이터가 입력으로 들어오는 것을 방지하기 위해 예외처리까지 해준다면 완벽한 코드가 됩니다.

```python
def normalize_defensive(inputs):
    if iter(inputs) is iter(inputs): # 일반 이터레이터는 거부합니다
        raise TypeError('Must supply a container')
    total = sum(inputs)
    result = []
    for value in inputs:
        percent = 100 * value / total
        result.append(percent)
    return result
```

`normalize_defensive` 함수는 이터레이터 프로토콜을 따르는 모든 컨테이터 타입에 대해 정상적으로 동작합니다. 하지만 컨테이너 타입이 아닌 일반 이터레이터는 예외를 일으킵니다.

```python
visits = [15, 35, 80]
normalize_defensive(visits) # 오류 없음
visits = ReadFile(path)
normalize_defensive(visits) # 오류 없음
it = iter(visits)
normalize_defensive(visits)

>>>
TypeError: Must supply a container
```

### 정리

- 리스트 컴프리헨션은 큰 입력을 처리할 때 메모리를 너무 많이 소비한다
- 제네레이터는 이터레이터를 통해 한 번에 한 출력만 내보내므로 메모리 문제를 피할 수 있다
- 제네레이터는 중첩이 가능하며 중첩될 때 매우 빠르게 실행된다
- 제네레이터를 사용하면 리스트를 여러번 사용하는 것 보다 이해하기 쉬운 코드를 작성할 수 있다
- 제네레이터는 모든 입출력 값을 메모리에 저장하지 않으므로 입력값의 양을 미리 알 수 없을때도 연속적인 출력을 만들 수 있다
- 함수의 입력 인수를 여러번 사용하는 경우 입력 인수가 이터레이터일 때 주의해야 한다
- 파이썬 이터레이터 프로토콜은 컨테이너와 이터레이터가 내장함수 `iter`, `next`, `for` 루프 및 관련 표현식과 상호작용 하는 방법을 정의한다
- `__iter__` 매직 메소드를 제네레이터로 구현하면 이터러블 컨테이너 타입을 쉽게 만들 수 있다
- 어떤 값에 iter를 두 번 호출 했을때 같은 결과가 나오고 내장함수 next로 전진시킬 수 있다면 그 값은 컨테이너가 아닌 이터레이터다
