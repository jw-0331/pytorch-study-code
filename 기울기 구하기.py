"""
x에 대한 기울기 구하기
"""

import torch

x=torch.tensor(data=[2.0,3.0], requires_grad=True)  #data: 배열, requires_grad: 이 텐서에 대한 기울기를 저장할지 여부 지정
y=x**2
z=2*y+3     #z=2x^2+3   z: 연산 그래프의 결과값

target=torch.tensor([3.0, 4.0])     #target: 목표값(정답)
loss=torch.sum(torch.abs(z-target))     #loss(오차)=(|결과값-목표값|합)  ->torch.sum하는 이유: 기울기(grad)는 스칼라에서만 구할 수 있기 때문
loss.backward()     #x에 대하여 기울기 구함

print(x.grad, y.grad, z.grad)       #x.grad=[8.0, 12.0], y.grad=None, z.grad=None =>x에 대하여만 기울기 저장할지를 지정했기 때문
#x.grad = 4*x = 4[2.0, 3.0] = [8.0, 12.0]
