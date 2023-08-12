# railroad
 2023 제 1회 철도 인공지능 경진대

https://aifactory.space/task/2511/overview

주행방식(5종류), 도로종류(2종류)에 따라 총 10종류의 데이터마다 0에서 3000미터 까지의 데이터가 있다. 라벨은 0에서 2500미터까지 있음.  
목표는 10종류의 모든 경우의 2501~3000까지의 탈선계수 예측하기.  
과거의 탈선계수를 반드시 사용할 것  
도로종류에 따라 조건이 바뀌므로 도로종류 상관없는 단일통합모델 또는 각 도로종류마다 모들을 2개 만들것  

RNN, transformer를 구현하며 시계열 데이터를 사용한 모델 구현을 목표로 대회에 참가


