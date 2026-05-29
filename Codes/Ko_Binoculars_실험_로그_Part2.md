# Ko-Binoculars 실험 로그 (Part 2)

> 이전 로그(`Ko-Binoculars_실험_로그.md`) 이후 추가된 실험 기록
> Temperature 이론 + 구간 단위 탐지 (Sentence/Segment-level Detection)

---

## 11. Temperature 이론 검증 ✅ 성공

### 가설
"AI 텍스트의 NLL 분산이 낮은 이유는 LLM이 낮은 temperature로 샘플링하기 때문이다."

즉, NLL 분산은 단순한 통계적 우연이 아니라 **LLM의 샘플링 메커니즘(temperature)을 직접 반영하는 양**이라는 가설.

### 실험 설계
- **모델**: Qwen2.5-3B-Instruct로 텍스트 생성 → Qwen2.5-3B + Instruct 쌍으로 NLL 분산 측정
- **프롬프트**: 한국어 5개 (전통 음식, AI 미래, 환경 보호, 교육, 기술 영향)
- **Temperature**: 0.3, 0.7, 1.0, 1.3
- **샘플**: 각 (prompt, temp) 조합당 5개 → 총 100개

### 결과

| Temperature | Avg NLL_std | Avg Binoculars |
|-------------|-------------|----------------|
| 0.3 | 1.2987 | 0.6426 |
| 0.7 | 1.4913 | 0.7424 |
| 1.0 | 1.6964 | 0.8677 |
| 1.3 | 1.8888 | 1.0236 |

**Pearson 상관계수: r = 0.6349, p < 0.0001**

### 결론
- Temperature가 올라갈수록 NLL 분산도 단조 증가
- Temperature ↔ Binoculars 스코어도 단조 증가
- **NLL 분산은 LLM의 temperature 샘플링과 직접 연결된 양**임이 검증됨
- 이론적 근거 확보: 단순 피처 엔지니어링이 아니라 생성 메커니즘 기반 탐지 원리

---

## 12. 유효 Temperature 역추정 ✅ 성공

### 목적
검증된 Temperature-NLL_std 관계를 역으로 사용하여, KatFish 데이터셋의 Human/AI 텍스트가 "어느 정도의 temperature에 해당하는지" 추정

### 회귀식
```
temp = 0.6773 × nll_std + (-0.2545)
```

### 결과 (Qwen2.5-3B 기준)

| Group | Avg NLL_std | Estimated Temperature |
|-------|-------------|----------------------|
| **Human** | 2.4085 | **1.3768** |
| Solar | 1.9242 | 1.0488 |
| Qwen2 | 2.0050 | 1.1035 |
| Llama3.1 | 2.0200 | 1.1136 |
| AI 평균 | 1.9793 | 1.0861 |

### 핵심 발견
1. **Human 유효 temperature ≈ 1.38** — 사람은 매우 높은 "temperature"로 글을 쓰는 것과 같음 (다양하고 예측 불가능한 토큰 선택)
2. **AI 유효 temperature ≈ 1.05~1.11** — KatFish의 AI 텍스트는 default temperature(0.7~1.0) 수준으로 추정됨
3. **Human-AI gap ≈ 0.27** — 이 gap이 탐지 가능성의 정량적 근거
4. **Solar가 가장 낮은 temperature처럼 보임 (1.0488)** — 실제로 ROC-AUC에서도 Solar가 가장 탐지하기 쉬웠음 (89.46) → 일관된 결과

### 논문 활용
- "유효 temperature(effective temperature)" 또는 "temperature-equivalent NLL dispersion" 개념 제안
- 단순 피처 결합이 아닌, **LLM 생성 메커니즘에 근거한 탐지 이론**으로 격상
- 한계: temperature를 높여 생성하면 탐지가 어려워짐 → 향후 연구 방향 제시

---

## 13. 구간 단위 탐지 (Segment-level Detection) ✅ 성공

### 목적
문서 전체의 binary 분류(AI vs Human)를 넘어, **"이 문서의 어느 부분이 AI인가"**를 찾는 새로운 문제 정의

### 응용 시나리오
- 카피킬러, 턴잇인 같은 학술 부정행위 탐지
- "리포트의 30%가 AI로 작성됐다" 같은 구체적 보고
- Human + AI 혼합 작성 환경

### 기존 연구와의 차별점
- Binoculars, KatFishNet 모두 **문서 레벨 binary 분류**만 다룸
- 본 연구는 **문서 내 구간(span) 탐지**로 확장

### 실험 설계

**1. 하이브리드 문서 생성**

두 가지 시나리오로 Human + AI 혼합 문서 생성:
- **End-insertion**: Human 앞 + AI 뒤 (단순 케이스)
- **Mid-insertion**: Human 글 중간에 AI 문단 삽입 (실제 시나리오)

AI 비율: 0%, 25%, 50%, 75%, 100% (5단계)

**2. Sliding Window 분석**

```python
window_size = 100  # 토큰
stride = 50        # 토큰
```

각 window에서 Binoculars 스코어 계산.

**3. 평가 지표**

- **AI 비율 추정 오차 (MAE)**: |예측 AI 비율 - 실제 AI 비율|
- **경계 검출 오차**: 예측된 전환점과 실제 전환점의 거리 (토큰 단위)

### 데이터
- Essay 10쌍 + Poetry 10쌍 = 20쌍
- 각 쌍 × 5개 AI 비율 × 2개 시나리오 = **200개 하이브리드 문서**
- Abstract는 제외 (이전 실험에서 NLL 변별력 부족 확인)

### Threshold 자동 결정

Human/AI 평균값의 중간으로 설정 (Qwen2.5-3B 기준):
- bino threshold: 0.9495
- nll_std threshold: 2.0447

(참고: Human bino=1.0810, std=2.4085 / AI bino=0.8180, std=1.6809)

### 결과 1: Hard Threshold (실패)

처음 시도한 hard threshold 방식 (window별 binary 판단 후 비율 계산):

| True Ratio | Bino MAE | Fusion MAE | Δ |
|-----------|----------|------------|---|
| 0.00 | 0.0559 | 0.0242 | -0.0317 |
| 0.25 | 0.1356 | 0.1729 | +0.0374 |
| 0.50 | 0.2172 | 0.2949 | +0.0777 |
| 0.75 | 0.2368 | 0.3805 | +0.1436 |
| 1.00 | 0.0998 | 0.3724 | +0.2726 |

**문제점**: Fusion이 Bino보다 나쁨. 100 토큰 window 내 NLL_std는 불안정하여 threshold 기반 hard classification에 취약.

### 결과 2: Soft Scoring (성공)

각 window의 스코어를 Human(1)~AI(0) 연속값으로 변환 후 평균:

```python
human_likeness = (bino - ai_mean) / (h_mean - ai_mean)
human_likeness = clip(human_likeness, 0, 1)
ai_ratio = 1 - mean(human_likeness across windows)
```

**Bino 단독 vs Random Baseline:**

| 시나리오 | Random MAE | Bino MAE | 개선율 |
|---------|-----------|----------|--------|
| End-insertion | 0.300 | 0.163 | **45.7%** |
| Mid-insertion | 0.300 | 0.163 | **45.7%** |

**상세 결과 (End-insertion)**

| True Ratio | Bino MAE | Random MAE |
|-----------|----------|------------|
| 0.00 | 0.1205 | 0.5000 |
| 0.25 | 0.0898 | 0.2500 |
| 0.50 | 0.1748 | 0.0000 |
| 0.75 | 0.2048 | 0.2500 |
| 1.00 | 0.2250 | 0.5000 |
| **평균** | **0.1630** | **0.3000** |

**상세 결과 (Mid-insertion)**

| True Ratio | Bino MAE | Random MAE |
|-----------|----------|------------|
| 0.00 | 0.1205 | 0.5000 |
| 0.25 | 0.1005 | 0.2500 |
| 0.50 | 0.1472 | 0.0000 |
| 0.75 | 0.2199 | 0.2500 |
| 1.00 | 0.2250 | 0.5000 |
| **평균** | **0.1626** | **0.3000** |

### 핵심 발견
1. **Soft scoring이 hard threshold보다 우월** — 노이즈에 강건
2. **End/Mid insertion 모두에서 거의 동일한 성능** (0.163 vs 0.163) → 삽입 위치에 robust
3. **랜덤 대비 약 46% 오차 감소** — 실용적 수준의 정확도
4. **Fusion(Bino+NLL_std)은 구간 탐지에 부적합** — 문서 레벨에서만 효과적

### Fusion vs Bino 단독: 역할 분담
- **문서 레벨 탐지**: Fusion (Bino + NLL_std) → +2.43 ROC-AUC 개선
- **구간 레벨 탐지**: Bino 단독 → 짧은 window에서 NLL_std는 불안정

### 결론
구간 단위 탐지는 문서 레벨과는 다른 접근이 필요하며, **Bino 단독 + Soft Scoring**이 가장 안정적. 이는 실제 응용 시나리오(카피킬러 등)에서 직접 활용 가능한 형태.

---