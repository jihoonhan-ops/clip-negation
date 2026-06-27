# Does CLIP Understand "No"? — Reproducing CLIP's Negation Blindness

CLIP 같은 vision-language model이 **부정(negation)**을 제대로 처리하지 못하는 현상을
작은 실험으로 재현하고 정량화한 프로젝트입니다.

> **동기**: 고려대 Vision & AI Lab의 ICCV 2025 논문
> *"Know 'No' Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP"*
> 를 읽고, CLIP의 negation 약점이 실제로 얼마나 심한지 직접 손으로 확인해보고 싶었습니다.

## 핵심 질문

CLIP은 `"a photo of a cat"`과 `"a photo of no cat"`을 구분할까?
직관적으로는 정반대 의미지만, CLIP의 텍스트 인코더는 `no` 같은 부정어를 거의 무시한다고 알려져 있습니다.

## 실험 설계

CIFAR-10 테스트 이미지를 사용해 세 가지를 측정합니다.

1. **표준 zero-shot baseline**
   `"a photo of a {class}"` 로 일반 분류 정확도 측정 → CLIP이 정상 작동함을 확인.

2. **Negation-ignored rate** (핵심 지표)
   분류 프롬프트를 `"a photo of no {class}"` 같은 **부정형**으로 바꿔서 분류.
   부정을 이해한다면 정답 클래스의 점수는 *낮아져야* 하지만,
   실제로는 여전히 정답 클래스가 1등으로 뽑히는 비율을 측정합니다.
   이 값이 높을수록 = CLIP이 `no`를 무시했다는 직접 증거.
   (부정 표현 4종을 비교: `no`, `with no`, `without`, `does not contain`)

3. **긍정 vs 부정 유사도 분포**
   같은 정답 이미지에 대해 `"a photo of a X"`와 `"a photo of no X"`의
   코사인 유사도 분포를 겹쳐 그립니다. 두 분포가 겹칠수록 부정을 무시한다는 뜻.

## 실행 방법

Google Colab에서 **GPU 런타임**으로:

```bash
pip install -r requirements.txt
python clip_negation.py --n_images 2000
```

처음 실행 시 CIFAR-10과 CLIP(ViT-B/32) 가중치를 자동으로 내려받습니다.
T4 GPU 기준 몇 분이면 끝납니다.

## 결과물

- `result_bar.png` — baseline 정확도 vs 부정 프롬프트별 negation-ignored rate
- `result_simdist.png` — 긍정/부정 프롬프트의 유사도 분포 비교
- 콘솔에 요약 지표 출력

## 해석 가이드 (README에 직접 채워 넣기)

실행 후 아래 표를 채우면 그대로 분석이 됩니다:

| 지표 | 값 |
|------|-----|
| 긍정 프롬프트 정확도 | __ % |
| 부정 프롬프트 평균 무시율 | __ % |
| 긍정-부정 유사도 평균 차이 | __ |

> 부정 무시율이 무작위(10%)보다 훨씬 높고 긍정 정확도에 가까울수록,
> CLIP이 부정어를 의미 있게 처리하지 못한다는 증거가 됩니다.

## 다음 단계 (확장 아이디어)

- 데이터셋을 **Oxford-IIIT Pets**(고화질, 37클래스)로 바꿔 더 선명한 결과 얻기
- 논문 방식대로 **부정 데이터를 추가 학습**시켜 약점이 줄어드는지 검증
- `"a photo of a cat and no dog"` 같은 **복합 부정** 프롬프트로 난이도 확장

## 참고

- Park et al., *Know "No" Better*, ICCV 2025 (Vision & AI Lab, Korea University)
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021
