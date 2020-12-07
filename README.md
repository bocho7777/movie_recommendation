# Movie Reccomendation 

# 영화 추천 시스템 웹서비스 구현하기 
##### 딥러닝 : tensorflow, 웹 서버: flask 프론트: Vue.js (API를 통한 연동)  

## <사용하는 추천 시스템 모델>
*기능은 상의 후 추가 예정 (user의 지난 영화를 토대로 추천하는거 user데이터 포함시킬지 여부) 


### **1) content-based filtering**

### - cosine similarity 를 이용한 유사 장르 추천시스템
       <flow>
       1. 자신이 본 영화를 선택한다. 
       2. cosine similarity를 이용 선택한 영화들과 유사한 장르의 영화를 추천한다.  
       
       
### - Bert 모델을 이용한 줄거리 기반 추천시스템 
      <flow>
      1. 자신이 선호하는 장르를 선택한다. 
      2. 선택한 장르에 해당하는 영화 데이터 중 즐겨본 데이터를 선택한다. 
      3. BERT 로 사용자가 입력한 값으로 사용자가 좋아할 것 같은 Movies를 출력한다. 
       
### **2) collaborative filtering**

### - 사용자 평점 유사도 기반 추천시스템 



## ML Web Service Architecture 


   ![Untitled Diagram (4)](https://user-images.githubusercontent.com/66239292/101338097-dcd83d00-38bf-11eb-8dbd-8dcc0577094d.png)



## 웹 서비스 와이어프레임 (1차) 

![dfd](https://user-images.githubusercontent.com/66239292/100518976-5a24f300-31d8-11eb-9e35-bd48b6c38181.PNG)


