

# Aiffel X Socar


<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=header&text=Aiffel X Socar&fontSize=90"/> 

#### 제 3차 해커톤 : AIFFELTHON



 > TEAM : **이음**




## <span style = "color: #2D3748; background-color:#fff5b1;"> 주제 : 차량 파손 탐지 </span>


### 1. 프로젝트 개요
+ 문제 제기
    - 현재 Socar 파손 탐지 모델은 classifier에서 정상/파손 이미지를 구분함
    - **파손과 비슷한 형태의 오염**이 segmentation 모델의 입력으로 사용될 가능성이 있음

<p align ='center'>
    <img src='./readme_image/socar_model_image.png' width='350px' alt>
</p>
<p align = 'center'>현재 SOCAR 파손 탐지 모델 </p>

</br>

+ 문제 정의
    - 파손과 유사한 오염/파손을 구분하는 **Classification** 문제
    - 이미지에서 파손 여부를 탐지하고, 파손 종류를 구분하는 **Segmentation** 문제

</br>

+ 해결 방안
    - **정상 / 파손 / 오염을 분류**하는 Classifier 사용
    - 분류된 파손 이미지를 **scratch / dent / spacing으로 파손 종류 및 영역 탐지**

</br>

### 2. 실험 절차 및 방법
<ul>
    <p align ='center'>
        <img src='./readme_image/flow.png' width='300px' alt>
    </p>
    <p align = 'center'>실험 절차 </p>
</ul>

+ Classifier 
    + 정상/오염/파손 이미지 분류


</br>

+ Segmentation
    + 파손 영역 탐지
        <p align ='center'>
            <img src='./readme_image/segmentation_flow.png' width='300px' alt>
        </p>
        <p align = 'center'> Segmentation 실험 절차 </p>
        
        </br>

        + U-Net 사용
            적은양의 데이터셋 에도 좋은 성능을 보여주는 특징으로 Baseline 모델로 선택
            <figure align = 'center'>
                <img src ='./readme_image/u_net.png' width='200px'>
                <figcaption> U-Net 구조 </caption>
            </figure>
        + 수행 과정
            1. Hyperparameter tuning
            2. Pretrained Backbone
            3. Fine-grained Backbone




### 3. 실험 결과
+ #### U-Net 하이퍼파라미터 튜닝 결과
<figure align ='center'>
    <img src ='./readme_image/unet-hypertuning.png' width='500px'></br>
    <figcaption> IOU Score 기준 최대 성능</figcaption>
</figure>


+ #### Backbone Model
    + 개선 아이디어
        - backbone 모델 사용
        - 다른 segmentation 모델 사용
        
            |모델||||
            |---|---|---|---|
            |U-Net| efficientnet b0 | efficientnet b2 | resnet50 |
            |DeeplabV3+| efficientnet b0 | efficientnet b2 | resnet50 |


+ ###### Backbone 모델 적용 결과
    + U-Net
    + DeeplabV3+

+ #### Fine-graine Backbone
    + 마스킹 이미지당 파손 비율

    + 개선 아이디어
        + **Fine-grained mlodel**로 학습된 backbone 사용

    + Fine grained model
        + DFL-CNN
            - 여러개의 feature map 사용
        + PMG
            - 이미지를 여러 크기의 패치 단위로 쪼개어 학습

+ ###### DFL-CNN
+ ###### PMG

### 4. 추후 연구 방향



</br>


<div align="center">
    <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat&logo=Pytorch&logoColor=white" />
	<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>
	<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=Jupyter&logoColor=white" />
</div>

- - -
###### [REEFERENCES]
- [쏘카 기술 블로그](https://tech.socarcorp.kr/data/2020/02/13/car-damage-segmentation-model.html#index3)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- [Learning a Discriminative Filter Bank within a CNN for Fine-grained Recognition](https://arxiv.org/pdf/1611.09932.pdf)
- [Fine-Grained Visual Classification via Progressive Multi-Granularity Training of Jigsaw Patches](https://arxiv.org/pdf/2003.03836.pdf)
- []