# 딥러닝을 이용한 개인정보 마스킹 촬영 시스템

## 주제 선정 배경 및 개발 목표
최근 여러가지 플랫폼에서 이루어지는 생방송 스트리밍이 굉장히 큰 인기를 얻고 있다. 누구나 촬영 기기만 있다면 언제 어디서든 실시간 방송이나 녹화 영상을 많은 사람들에게 제공할 수 있다. 하지만 이에 따라 생방송 스트리밍 중 당사자의 동의 없이 얼굴이 노출되는 등의 초상권 침해와 관련된 개인 정보 노출 문제가 많이 발생하고 있다. 스트리머 입장에서도 해당 문제에 대해 단순히 방송인을 제외한 얼굴이 화면에 나오지 않도록 조심히 촬영하거나 같은 공간에 있는 모든 사람에게 촬영 동의를 구하는 비현실적인 방법 외에는 특별한 해결책이 없다. 이 문제를 해결하기 위해 딥러닝 얼굴인식 기술을 이용하여 실시간 영상에서 사전에 등록되지 않은 얼굴을 마스킹 처리하는 방식을 적용한 작품을 개발한다.

얼굴검출과 얼굴인식 딥러닝 모델을 이용하여 실시간 영상 프레임에서 얼굴을 검출하고 사전 등록된 얼굴과 비교해 분류해내어 마스킹 처리를 실시하는 알고리즘과 스마트폰에서 촬영하는 영상을 서버로 전송하고 처리된 영상을 웹 페이지에서 실시간으로 시청할 수 있는 프로젝트를 개발한다. 이로써 실시간 영상에 자동으로 마스킹 처리가 적용되어 실시간 스트리밍 중 초상권 침해 등의 문제를 자동으로 해결할 수 있는 시스템 구축을 목표로 한다.

## 시스템 시나리오
![image](https://user-images.githubusercontent.com/96522336/173508476-b597a2fe-6a52-4c09-9739-620ed3d14fcd.png)

## 시스템 구성도
![image](https://user-images.githubusercontent.com/96522336/173508593-4ef5c0ef-fa4f-4c82-8ab9-c8c38d829592.png)

## 전체 흐름도
![image](https://user-images.githubusercontent.com/96522336/173508646-2c9cc985-f459-44ff-acd8-5864fb19f061.png)

## 필요 기술 및 참고 문헌
FaceNet
+ A Unified Embedding for Face Recognition and Clustering
+ https://github.com/davidsandberg/facenet
+ https://arxiv.org/abs/1503.03832

OpenCV
+ https://opencv.org

RTMP(Real Time Messaging Protocol)

기타 인공지능, 네트워크, 코틀린, Django, DB서적

## 팀원 구성
이승준 https://github.com/Shlaom

김강섭 https://github.com/eneleh758 

윤예진 https://github.com/yejin603
