# tensorflow 기초다지기
## 이 문서는 [텐서플로우 블로그](https://tensorflow.blog/1-%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EA%B8%B0%EB%B3%B8%EB%8B%A4%EC%A7%80%EA%B8%B0-first-contact-with-tensorflow/) 를 참조하여 작성되었습니다.(사실상 복붙..)
텐서플로우는 데이터 플로우 그래프를 사용해서 수치 연산을 하는 라이브러리로 볼 수 있습니다. 그래프의 노드(node)는 수학적 연산을 나타내고 노드를 연결하는 그래프의 엣지(edge)는 다차원 데이터 배열(array)을 나타냅니다.
또한, 텐서플로우는 수치연산을 기호로 표현한 그래프 구조를 만들고 처리한다는 기본 아이디어를 바탕으로 구현되었습니다. 
<br/>

그래서 텐서플로우는 [CPU, GPU](https://github.com/Kim-Jiwan/N.S.T-project/blob/master/study/CPUvsGPU.md) 의 장점을 모두 이용할 수 있고 안드로이드나 iOS 같은 모바일 플랫폼은 물론 맥 OS X와 같은 64비트 리눅스에서 바로 사용될 수 있습니다.
텐서플로우의 또 하나의 강점은 알고리즘이 어떻게 돌아가고 있는지 알려주기 위해 많은 정보를 모니터링하고 디스플레이 해주는 텐서보드 모듈입니다. 좀 더 좋은 모델을 만들기 위해서 알고리즘의 동작을 조사해서 디스플레이 하는 것이 매우 중요합니다. 종종 시행착오를 통해 약간 불분명한 프로세스로 많은 모델을 만들고 있는데 이건 당연히 리소스 특히 시간 낭비입니다.

* ## tensorflow 설치
tensorflow의 파이썬 API를 사용하려면 파이썬 2.7 버전을 설치해야 합니다.(특별한 이유가 없다면 3.5.x 버전 이상의 파이썬을 사용하는 것이 좋습니다.)
<br/>

일반적으로 파이썬으로 작업을 할 때는 virtualenv라는 가상환경을 사용해야 합니다. virtualenv는 한 컴퓨터에서 여러 프로젝트를 작업할 때 파이썬 패키지의 의존성이 충돌하지 않도록 관리해주는 툴 입니다. 즉 virtualenv를 사용하여 텐서플로우를 설치하면 의존성 때문에 같이 설치 되는 패키지들이 다른 프로젝트에서 설치한 같은 패키지들을 덮어쓰지 않게 됩니다.
<br/>

우선 pip와 virtualenv를 아래 명령으로 설치합니다.
```bash
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev python-virtualenv 

# Mac OS X 
$ sudo easy_install pip
$ sudo pip install --upgrade virtualenv
```
~/tensorflow 디렉토리에 virtualenv 환경을 만듭니다.
```bash
$ virtualenv --system-site-packages ~/tensorflow
```
다음은 아래처럼 virtualenv를 활성화시키는 것입니다.
```bash
$ source ~/tensorflow/bin/activate #  with bash 
$ source ~/tensorflow/bin/activate.csh #  with csh
tensorflow)$
```
활성화된 후에는 명령줄 시작 부분에 현재 작업하고 있는 virtualenv 이름이 나타나게 됩니다. virtualenv가 활성화 되었으므로 pip를 이용해 텐서플로우를 설치합니다.
```bash
# Ubuntu/Linux 64-bit, CPU only:
(tensorflow)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl 

# Mac OS X, CPU only:
(tensorflow)$ sudo easy_install --upgrade six
(tensorflow)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
```
최신 버전(Python3.5 ~ 3.7)을 설치하려면 공신 설치 문서인 [여기](https://www.tensorflow.org/install/) 를 참고하세요.
<hr/>

* ## 첫 번째 tensorflow 코드
텐서플로우 프로그램이 어떤지 처음 맛보기 위해 간단한 곱셈 프로그램을 만들어 보겠습니다. 코드는 아래와 같습니다.

```tensorflow
import tensorflow as tf

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b)

sess = tf.Session()

print sess.run(y, feed_dict={a: 3, b: 3})
```
이 코드에서 텐서플로우 파이썬 모듈을 임포트한 후 프로그램 실행 중에 값을 변경할 수 있는 placeholder라 부르는 심볼릭 변수들을 정의합니다. 그리고 나서 텐서플로우에서 제공하는 곱셈 함수를 호출할 때 이 두 변수를 파라메타로 넘깁니다. tf.multiply은 텐서(tensor)를 조작하기 위해 텐서플로우가 제공하는 많은 수학 연산 함수 중 하나입니다. 여기서 텐서는 동적 사이즈를 갖는 다차원 데이터 배열이라고 생각하면 됩니다.
<br/>

주요한 함수는 아래와 같습니다. (텐서플로우에는 이외에도 많은 기본적인 수학함수를 제공하고 있습니다. 상세한 내용은[API 문서](https://www.tensorflow.org/guide) 를 참고해 주세요.)
<br/>

함수|설명| -
---|:---:|---
tf.add| 덧셈 |
tf.subtract| 뺄셈 |
tf.multiply| 곱셈 |
tf.div| 나눗셈의 몫(Python 2 스타일) |
tf.truediv| 나눗셈의 몫(Python 3 스타일) |
tf.mod| 나눗셈의 나머지 |
tf.abs| 절댓값을 리턴 |
tf.negative| 음수를 리턴 |
tf.sign| 부호를 리턴(음수는 -1, 양수는 1, 0은 0을 리턴) |
tf.reciprocal| 역수를 리턴(3의 역수는 1/3) |
tf.square| 제곱을 계산 |
tf.round| 반올림 값을 리턴 |
tf.sqrt| 제곱근을 계산 |
tf.pow| 거듭제곱 값을 계산 |
tf.exp| 지수 값을 계산 |
tf.log| 로그 값을 계산 |