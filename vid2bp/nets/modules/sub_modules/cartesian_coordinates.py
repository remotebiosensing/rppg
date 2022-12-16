'''
#TODO:
노트 p.30
1. 하나의 cycle을 찾음.
2. cycle을 다항식으로 근사화 ( x^4 + 2x^3 + 3x^2 + 4x + 5 )
3. 근사화된 다항식에서 표준기저를 찾음 reversed( x^4, x^3, x^2, x, 1 ) / ( 또는 표준 벡터 생성 (1, 1, 1, 1, 1) )
4. 표준기저에 따른 class 분류하는 모델 생성

1 ~ 4를 통해 ppg의 cycle을 분류하는 모델 생성하여 실제 abp prediction을 하기 전에 어떤 abp cycle을 가지는지 힌트를 제공


노트 p.33
1. 전체 데이터를 2차원으로 변환 (how?)
2. eigenvalue , eigenvector를 구함
3. ppg의 eigenvector에서 abp의 eigenvector로 변환하도록 모델 구성


노트 p.53
미분방정식?

느낀점:
 지금까지 수학 문제 푼 것들은 그 방식대로 계산이 가능한 case들만 보고 풀어왔구나를 느낌. 즉, 수학적으로 풀 수 있는 문제들만 풀어왔다는 것을 느낌.
 어떻게 모델이 풀 수 있는 문제를 만들어 줄 수 있을까?
'''

import numpy as np
import matplotlib.pyplot as plt

# def dimension_expansion(input_sig, degree=1):

