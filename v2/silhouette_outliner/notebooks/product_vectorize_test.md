코딩 에이전트(AI Agent)가 데이터 파이프라인 설계, 전처리, 모델링, 그리고 검증 전략까지 명확하게 이해하고 코드를 작성할 수 있도록 기술 사양서(Technical Specification) 형태로 정리한 문서입니다.

마크다운(Markdown) 포맷으로 작성되었으니 그대로 복사해서 에이전트의 컨텍스트(프롬프트)나 `README.md`로 제공하시면 됩니다.

---

# [Technical Specification] Musinsa Ranking Analysis: Input Space vs. Latent Space (Metric Learning)

## 1. Overview & Objective

본 프로젝트의 목적은 무신사의 제품 속성(Features)과 랭킹(Ranking) 간의 상관관계를 분석하고, 이를 활용해 순위 변동을 예측하거나 제품의 경쟁력을 평가하는 모듈을 개발하는 것이다.
코딩 에이전트는 아래 정의된 두 가지 방법론을 독립적인 모듈로 구현하고, 노트북 레벨에서 테스트 및 평가한 후 최종 기능 모듈로 고도화한다.

---

## 2. Target Dataset Definition (Data Schema)

모든 모델의 입력 데이터 프레임(`df`)은 상품별로 다음 변수들을 포함해야 한다.

* **종속 변수 (Target, $Y$):** `delta_ranking` (어제 순위 - 오늘 순위, 연속형 수치)
* **독립 변수 (Features, $X$):**
* Continuous: `price` (가격), `discount_rate` (할인율), `review_count` (리뷰 수), `like_count` (좋아요 수)
* Categorical: `is_domestic` (도메스틱 여부: 0 또는 1), `style_tag` (원핫 인코딩 대상)



---

## 3. Approach A: Physical Input Space Vectorization (속성 공간 분석)

### 3.1 개념 및 작동 원리

수집된 $N$개의 제품 속성을 각각의 독립된 차원으로 취급하여 $N$차원의 물리적 공간을 구성한다. 제품은 이 공간 위의 하나의 점(Vector)으로 표현되며, 제품 간의 순수한 기하학적 유사도를 측정한다.

### 3.2 파이프라인 및 에이전트 구현 지침

1. **Preprocessing:**
* `review_count`, `like_count` 등 편차가 큰 수치형 변수는 `np.log1p()`로 로그 변환을 수행할 것.
* `StandardScaler`를 사용하여 모든 연속형 변수의 스케일을 평균 0, 표준편차 1로 동기화할 것 (유클리디안 거리 왜곡 방지).


2. **Distance Metric:** `scipy.spatial.distance.cdist` 또는 `sklearn.metrics.pairwise_distances`를 사용하여 코사인 유사도(Cosine Similarity) 또는 유클리디안 거리를 계산할 것.
3. **Statistical Validation (노트북 테스트 필수 요구사항):**
* 거리 기반으로 정렬된 이웃 제품들 간의 `delta_ranking` 상관관계($r$)를 Pearson 또는 Spearman 기법으로 측정하여 "스펙이 유사하면 순위 변동도 유사한가?"를 검증할 것.



### 3.3 설명성 (Explainability)

* **장점:** 물리적 축(가격, 리뷰 등)이 유지되므로 "두 제품은 가격이 비슷하고 리뷰 수가 비슷해서 가깝다"라는 변수 단위의 직관적 설명 가능.
* **단점:** 랭킹 알고리즘의 비선형적 가중치나 상호작용(Interaction)을 반영하지 못함.

---

## 4. Approach B: Metric Learning Latent Space (잠재 공간 분석)

### 4.1 개념 및 작동 원리

입력 속성 벡터를 인공신경망(MLP)에 통과시켜 차원을 변경하되, **"무신사 랭킹/순위 변동이 유사한 제품들은 가깝게, 순위 차이가 큰 제품들은 멀어지도록"** 공간의 축 자체를 비틀어 학습시키는 방법론이다.

### 4.2 파이프라인 및 에이전트 구현 지침

1. **Data Loader (Triplet Generator):**
* 데이터셋을 `[Anchor, Positive, Negative]`의 Triplet 형태로 재구성하는 커스텀 데이터로더를 구현할 것.
* `Anchor`: 기준 제품
* `Positive`: Anchor와 `delta_ranking` 차이가 정해진 임계값(Threshold) 이내인 제품
* `Negative`: Anchor와 `delta_ranking` 차이가 인위적으로 큰 제품


2. **Model Architecture:** PyTorch 또는 TensorFlow를 활용하여 2~3개 레이어의 간단한 MLP(Embedding Network)를 빌드할 것.
3. **Loss Function:** `nn.TripletMarginLoss`를 활용하여 $d(A, P)$는 최소화하고, $d(A, N)$은 최대화하도록 가중치를 학습시킬 것.
4. **Evaluation & Visualization (노트북 테스트 필수 요구사항):**
* 학습된 잠재 벡터들을 **UMAP** 또는 **t-SNE**를 통해 2차원으로 축소하여 시각화할 것.
* 시각화 시 점의 색상을 `delta_ranking` 또는 현재 순위로 칠했을 때, 공간상에 순위별 그라데이션(Clustering)이 형성되는지 확인할 것.



### 4.3 설명성 (Explainability)

* **장점:** 무신사 랭킹 시스템의 '숨은 가중치와 메커니즘'을 다차원 공간의 거리로 표현하므로, 포텐셜 있는 제품 추적 및 성장을 위한 방향성(Vector Direction) 제시 가능.
* **단점:** 변수들이 블랙박스(MLP)를 거치며 뒤섞이므로 특정 축이 어떤 물리적 의미를 갖는지 1:1 매칭이 불가능함 (추후 SHAP 등의 라이브러리로 보완 필요).

---

## 5. Evaluation Protocol & Next Action (에이전트 태스크 순서)

에이전트는 단계적으로 다음 작업을 수행하며 코드를 빌드업해야 한다.

1. **Phase 1 (Data & Preprocessing):** 무신사 더미 데이터를 생성하거나 로드하여 수치형/범주형 전처리 파이프라인 함수를 모듈화한다.
2. **Phase 2 (Approach A Sandbox):** 노트북 환경에서 Approach A를 구현하고 이웃 기반의 순위 상관계수를 도출한다.
3. **Phase 3 (Approach B Sandbox):** PyTorch 기반 Triplet Loss 모델을 구현하고 학습 전/후의 UMAP 시각화 대조 플롯을 생성한다.
4. **Phase 4 (Refactoring):** Sandbox 코드를 구조화하여 `src/models/input_space_analyzer.py`, `src/models/metric_learner.py` 등의 프로덕션 급 기능 모듈로 패키징한다.

---

수정이나 추가가 필요한 서약(Constraints)이나 특정 타깃 라이브러리(PyTorch, Scikit-learn 등)가 있다면 에이전트에게 전달하기 전에 문서에 덧붙여 주시면 됩니다.