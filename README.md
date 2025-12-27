# 📊 Event-Driven Trading System with Meta-Filtering & Rule-Based Strategy

## 📌 프로젝트 개요
본 프로젝트는 **이벤트 기반 트레이딩 시스템**을 구축하는 것을 목표로 합니다.  
뉴스, 텔레그램 채널 등에서 발생하는 다양한 이벤트를 기반으로 가격 반응을 분석하고,  
메타-필터링과 Rule-Based 전략을 통해 실제 트레이딩에 활용 가능한 시그널을 만드는 시스템입니다.

---

## ✅ 1. 지금까지 구현한 것 (Implemented)

### 🔹 이벤트 기반 트레이딩 이론 정리
- Event-Driven Trading 구조 정의
- 정보 흐름과 가격 반응 관계 분석
- 수익 창출 메커니즘 이론화

### 🔹 Triple Barrier 적용 및 레이블링
- Triple Barrier 개념 정리
- 상·하 Barrier 및 최대 보유 기간 설정
- 이벤트 이후 가격 반응 기반 **TP / FP / TN / FN 분류**

### 🔹 기본 신호 평가 체계 구축
- 이벤트 발생 후 수익/손실 여부 판단 구조 구성
- 단순 수익률이 아닌 **결과 기반 레이블링 체계 수립**

### 🔹 노트북 기반 연구 정리
- `순수한 이론을 따라가보자.ipynb` 작성 및 정리
- 전략 아이디어 및 실험 방향 기록
- Triple Barrier 기반 실험 구조 초안 완성

---

## 🔄 2. 현재 진행 중인 것 (In Progress)

### 📡 텔레그램 기반 Meta-Filtering 시스템 개발
- 텔레그램 채널로부터 데이터 수집
- 채널 신뢰도 기반 가중치 적용 설계
- 중복/스팸/낮은 품질 신호 제거
- 중요 이벤트 자동 추출 알고리즘 연구

### ⏱ 1분봉 기반 이벤트 반응 분석
- 1-minute candle 기준 초단기 반응 분석
- 더 많은 이벤트 탐지(High Recall) 실험
- 미세 변동성 반응 모델링
- TP / FP 분류 정확도 향상

### 🧠 TP / FP 분류 고도화
- 단순 라벨링 → 실제 분류 모델로 확장
- 이벤트의 “품질”에 대한 판단 기준 설계
- False Positive 감소를 위한 추가 필터링 연구

---

## 🚀 3. 앞으로 구현할 예정인 것 (Planned / TODO)

### 🤖 자동 트레이딩 봇 구현
- 이벤트 수집 → 필터링 → 시그널 생성 → 주문
- 실시간 파이프라인 구축
- 백테스트 + 실거래 구조 설계

### 📈 Rule-Based 전략 고도화
- 단순 Triple Barrier 수익여부 판별을 넘어서
- **추세 추종(Trend Following)** 기반 Rule 강화
- 시장 구조 기반 조건식 설계
- Rule Engine 형태로 모듈화

### 🧩 Event + Rule 융합 전략
- Event-Driven + Trend-Following 결합
- “설명 가능성 있는 전략(Explainable Strategy)” 지향
- 단순 ML이 아닌, 규칙 기반 구조 유지

---

## 🛠 사용 기술 스택 (현재 & 예정)
- Python
- Pandas / Numpy
- Scikit-learn
- Telegram API
- Event Detection Pipeline
- Triple Barrier Labeling
- (예정) Trading API
- (예정) 백테스트 프레임워크

---

## 📎 연구 노트북
- `순수한 이론을 따라가보자.ipynb`
  - 이벤트 트레이딩 이론
  - Triple Barrier 전략
  - 메타-필터링 개념 정리
  - 전략 설계 과정 기록

---

⚠ **Disclaimer**  
본 프로젝트는 연구 및 학습 목적입니다.  
투자 권유가 아니며 모든 투자 책임은 본인에게 있습니다.

