"""
NoLiMa 논문 기준으로 현재 NIAH 데이터셋의 문제점을 심층 분석
"""
import json
import re
from collections import Counter
import numpy as np

def analyze_question_diversity(samples):
    """질문 다양성 분석"""
    questions = [s['question'] for s in samples]
    unique_questions = set(questions)

    return {
        'total_samples': len(samples),
        'unique_questions': len(unique_questions),
        'diversity_ratio': len(unique_questions) / len(samples),
        'most_common': Counter(questions).most_common(5)
    }

def analyze_needle_pattern(samples):
    """Needle 패턴 분석"""
    patterns = []
    for sample in samples:
        context = sample['context']
        answer = sample['answer']

        # Needle 찾기
        needle_pattern = rf"[Tt]he secret passkey is {answer}"
        match = re.search(needle_pattern, context, re.IGNORECASE)

        if match:
            # Needle 전후 50자 추출
            start = max(0, match.start() - 50)
            end = min(len(context), match.end() + 50)
            patterns.append(context[start:end])

    return patterns[:10]  # 처음 10개만

def analyze_context_repetition(sample):
    """문맥 내 문장 반복도 분석"""
    context = sample['context']
    sentences = [s.strip() for s in re.split(r'[.!?]+', context) if s.strip()]

    # 각 문장의 등장 횟수
    sentence_counts = Counter(sentences)
    repeated_sentences = {s: c for s, c in sentence_counts.items() if c > 1}

    # 가장 많이 반복된 문장들
    most_repeated = sentence_counts.most_common(5)

    total_sentences = len(sentences)
    unique_sentences = len(set(sentences))

    return {
        'total_sentences': total_sentences,
        'unique_sentences': unique_sentences,
        'repetition_ratio': (total_sentences - unique_sentences) / total_sentences if total_sentences > 0 else 0,
        'most_repeated': most_repeated,
        'num_repeated_types': len(repeated_sentences)
    }

def simulate_keyword_matching(sample):
    """키워드 매칭만으로 정답을 찾을 수 있는지 시뮬레이션"""
    question = sample['question'].lower()
    context = sample['context'].lower()
    answer = sample['answer'].lower()

    # 질문에서 키워드 추출
    keywords = set(re.findall(r'\b\w+\b', question)) - {'the', 'is', 'in', 'what', 'a', 'an', 'to', 'of', 'and'}

    # "secret passkey" 주변 찾기
    if 'secret' in keywords and 'passkey' in keywords:
        # "secret passkey is" 패턴 찾기
        pattern = r'secret\s+passkey\s+is\s+([A-Z0-9]{6})'
        match = re.search(pattern, context, re.IGNORECASE)

        if match:
            predicted = match.group(1).lower()
            return predicted == answer

    return False

def calculate_nolima_score(samples):
    """NoLiMa 기준 점수 계산 (0-100)"""
    scores = {
        'question_diversity': 0,  # 질문 다양성 (30점)
        'lexical_decoupling': 0,  # 어휘 분리도 (30점)
        'context_complexity': 0,  # 문맥 복잡도 (20점)
        'pattern_generalization': 0  # 패턴 일반화 (20점)
    }

    # 1. 질문 다양성
    q_analysis = analyze_question_diversity(samples)
    scores['question_diversity'] = q_analysis['diversity_ratio'] * 30

    # 2. 어휘 분리도 (키워드 매칭으로 해결 불가)
    keyword_solvable = sum(1 for s in samples[:50] if simulate_keyword_matching(s))
    scores['lexical_decoupling'] = (1 - keyword_solvable/50) * 30

    # 3. 문맥 복잡도 (반복이 적을수록 높음)
    rep_ratios = [analyze_context_repetition(s)['repetition_ratio'] for s in samples[:20]]
    avg_rep = np.mean(rep_ratios)
    scores['context_complexity'] = (1 - avg_rep) * 20

    # 4. 패턴 일반화 (Needle 패턴이 다양할수록 높음)
    patterns = analyze_needle_pattern(samples[:20])
    unique_patterns = len(set(patterns))
    scores['pattern_generalization'] = (unique_patterns / len(patterns)) * 20 if patterns else 0

    return scores

def main():
    # Load datasets
    with open('data/processed/niah/global_niah.jsonl') as f:
        global_data = [json.loads(line) for line in f]

    with open('data/processed/niah/korean_niah.jsonl') as f:
        korean_data = [json.loads(line) for line in f]

    for name, data in [("Global", global_data), ("Korean", korean_data)]:
        print(f"\n{'='*80}")
        print(f"📋 {name} NIAH - NoLiMa 준수도 분석")
        print(f"{'='*80}")

        # 1. 질문 다양성
        print(f"\n🎯 1. 질문 다양성")
        q_div = analyze_question_diversity(data)
        print(f"  총 샘플: {q_div['total_samples']}개")
        print(f"  고유 질문: {q_div['unique_questions']}개")
        print(f"  다양성 비율: {q_div['diversity_ratio']*100:.1f}%")
        print(f"  가장 빈번한 질문 (상위 3개):")
        for q, count in q_div['most_common'][:3]:
            print(f"    \"{q[:60]}...\" - {count}회")

        # 2. Needle 패턴
        print(f"\n🔍 2. Needle 패턴 분석")
        patterns = analyze_needle_pattern(data)
        print(f"  샘플 패턴 (처음 3개):")
        for i, p in enumerate(patterns[:3]):
            print(f"    #{i+1}: ...{p}...")

        # 3. 문맥 반복도
        print(f"\n📊 3. 문맥 반복도 분석 (샘플 1개)")
        rep_analysis = analyze_context_repetition(data[0])
        print(f"  총 문장 수: {rep_analysis['total_sentences']}개")
        print(f"  고유 문장 수: {rep_analysis['unique_sentences']}개")
        print(f"  반복 비율: {rep_analysis['repetition_ratio']*100:.1f}%")
        print(f"  가장 많이 반복된 문장 (상위 3개):")
        for sent, count in rep_analysis['most_repeated'][:3]:
            print(f"    {count}회: \"{sent[:70]}...\"")

        # 4. 키워드 매칭 시뮬레이션
        print(f"\n🤖 4. 키워드 매칭 시뮬레이션 (처음 50개)")
        keyword_solved = sum(1 for s in data[:50] if simulate_keyword_matching(s))
        print(f"  키워드 매칭만으로 해결 가능: {keyword_solved}/50 ({keyword_solved/50*100:.0f}%)")

        # 5. NoLiMa 점수
        print(f"\n🎓 5. NoLiMa 준수도 점수")
        scores = calculate_nolima_score(data)
        total_score = sum(scores.values())
        print(f"  질문 다양성 (30점 만점): {scores['question_diversity']:.1f}점")
        print(f"  어휘 분리도 (30점 만점): {scores['lexical_decoupling']:.1f}점")
        print(f"  문맥 복잡도 (20점 만점): {scores['context_complexity']:.1f}점")
        print(f"  패턴 일반화 (20점 만점): {scores['pattern_generalization']:.1f}점")
        print(f"  \n  📈 총점: {total_score:.1f}/100점")

        # 6. NoLiMa 개선 권장사항
        print(f"\n💡 6. NoLiMa 기준 개선 권장사항")
        recommendations = []

        if scores['question_diversity'] < 15:
            recommendations.append("❌ 질문을 다양화하세요 (현재 거의 동일)")

        if scores['lexical_decoupling'] < 15:
            recommendations.append("❌ 질문과 needle의 어휘 중복을 줄이세요")

        if scores['context_complexity'] < 10:
            recommendations.append("❌ 문맥의 반복을 줄이고 다양성을 높이세요")

        if scores['pattern_generalization'] < 10:
            recommendations.append("❌ Needle 표현 방식을 다양화하세요")

        if recommendations:
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print(f"  ✅ 주요 개선사항 없음")

if __name__ == "__main__":
    main()
