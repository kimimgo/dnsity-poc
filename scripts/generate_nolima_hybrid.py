"""
NoLiMa 스타일 NIAH 데이터셋 생성 - 하이브리드 방식
- 규칙 기반: 50개 (완벽한 어휘 분리 보장)
- LLM 생성: 150개 (Gemini CLI 활용, 다양성 확보)
"""
import json
import random
import string
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# ============================================================================
# 1. 유틸리티 함수
# ============================================================================

def generate_code(length: int = 6) -> str:
    """6자리 랜덤 코드 생성"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def calculate_lexical_overlap(text1: str, text2: str) -> float:
    """두 텍스트 간 어휘 중복률 계산"""
    stopwords = {'the', 'is', 'in', 'what', 'a', 'an', 'to', 'of', 'and', 'for', 'that', 'this', 'are', 'was', 'be'}
    words1 = set(re.findall(r'\b\w+\b', text1.lower())) - stopwords
    words2 = set(re.findall(r'\b\w+\b', text2.lower())) - stopwords

    if not words1:
        return 0.0

    overlap = len(words1 & words2)
    return overlap / len(words1)

def calculate_ttr(text: str) -> float:
    """Type-Token Ratio 계산"""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def validate_sample(sample: Dict) -> Tuple[bool, str]:
    """NoLiMa 품질 검증"""
    # 1. Lexical overlap 검사
    overlap = calculate_lexical_overlap(sample['question'], sample['needle_phrase'])
    if overlap > 0.1:
        return False, f"Overlap too high: {overlap:.2f}"

    # 2. TTR 검사
    ttr = calculate_ttr(sample['context'])
    if ttr < 0.2:
        return False, f"TTR too low: {ttr:.3f}"

    # 3. 코드 형식 검사
    if not re.match(r'^[A-Z0-9]{6}$', sample['answer']):
        return False, f"Invalid code format: {sample['answer']}"

    # 4. Needle이 context에 포함되어 있는지
    if sample['answer'] not in sample['context']:
        return False, "Answer not in context"

    return True, "OK"

# ============================================================================
# 2. Haystack 템플릿 (다양한 도메인)
# ============================================================================

HAYSTACK_PARAGRAPHS = {
    "technology": [
        "Cloud computing infrastructure enables scalable deployment of enterprise applications across distributed systems. Container orchestration platforms streamline development workflows and reduce operational overhead. Microservices architecture promotes modular design principles and independent service scaling. DevOps practices integrate development and operations teams for faster delivery cycles.",
        "Machine learning algorithms process vast datasets to identify patterns and generate predictions. Neural network architectures have evolved to handle increasingly complex computational tasks. Natural language understanding systems enable human-computer interaction through conversational interfaces. Computer vision applications transform image data into actionable insights for various industries.",
        "Cybersecurity frameworks establish protocols for protecting sensitive information assets. Encryption standards ensure data confidentiality during transmission and storage. Identity management systems verify user credentials across multiple platforms. Threat detection mechanisms monitor network activity for suspicious behavior patterns.",
    ],
    "business": [
        "Strategic planning initiatives align organizational objectives with market opportunities. Competitive analysis frameworks evaluate industry dynamics and positioning strategies. Resource allocation decisions optimize operational efficiency and return on investment. Performance metrics track progress toward quarterly and annual targets.",
        "Customer relationship management systems centralize client interaction data for analysis. Sales pipeline automation streamlines lead qualification and conversion processes. Marketing campaigns leverage data analytics for targeted audience engagement. Brand positioning strategies differentiate offerings in crowded marketplaces.",
        "Financial reporting standards ensure transparency in corporate disclosures and communications. Regulatory compliance requirements govern operational procedures across jurisdictions. Risk assessment methodologies quantify potential exposure to adverse events. Audit procedures verify accuracy of financial statements and internal controls.",
    ],
    "healthcare": [
        "Clinical decision support systems assist physicians with diagnostic recommendations. Electronic medical records facilitate information sharing across healthcare providers. Telemedicine platforms expand access to specialist consultations in remote areas. Patient safety protocols minimize risks during medical procedures and treatments.",
        "Pharmaceutical research follows rigorous testing protocols before regulatory approval. Drug interaction databases help clinicians avoid adverse medication combinations. Personalized medicine approaches tailor treatments based on genetic profiles. Clinical trial management systems coordinate participant enrollment and data collection.",
        "Population health analytics identify trends in disease prevalence and risk factors. Preventive care programs reduce long-term healthcare costs through early intervention. Mental health services integrate with primary care for comprehensive treatment. Rehabilitation protocols guide recovery from injuries and surgical procedures.",
    ],
    "legal": [
        "Contract negotiations establish terms and conditions governing business relationships. Intellectual property protections secure innovations through patents and trademarks. Dispute resolution mechanisms provide alternatives to traditional litigation. Compliance monitoring ensures adherence to regulatory requirements across operations.",
        "Corporate governance frameworks define roles and responsibilities of board members. Merger and acquisition due diligence evaluates financial and legal aspects of transactions. Employment law regulations govern workplace policies and employee rights. Data privacy legislation mandates handling procedures for personal information.",
        "Environmental regulations impose standards for sustainable business practices. Consumer protection laws safeguard buyers from deceptive marketing practices. International trade agreements facilitate cross-border commercial activities. Antitrust enforcement prevents monopolistic behavior in competitive markets.",
    ],
    "finance": [
        "Investment portfolio diversification strategies balance risk and return objectives. Asset allocation models distribute capital across different security classes. Market volatility indices measure fluctuations in financial instrument prices. Quantitative trading algorithms execute transactions based on statistical models.",
        "Banking regulations establish capital requirements for financial institutions. Credit risk assessment frameworks evaluate borrower default probability. Liquidity management ensures sufficient cash reserves for operational needs. Interest rate derivatives hedge exposure to monetary policy changes.",
        "Insurance underwriting processes evaluate risk factors for policy pricing. Claims processing workflows verify coverage and calculate settlement amounts. Actuarial models project future liabilities based on historical data. Reinsurance arrangements distribute risk among multiple insurance providers.",
    ],
}

# ============================================================================
# 3. NoLiMa 질문-Needle 페어 (어휘 분리 보장)
# ============================================================================

# NoLiMa 핵심: 질문과 needle에 공통 단어가 없어야 함
# 각 템플릿에서 질문/needle의 단어가 완전히 분리되도록 설계
NOLIMA_TEMPLATES = {
    "vault": [
        {
            "needle": "The secure facility requires token: {CODE}",
            "questions": [
                "What alphanumeric string unlocks the protected area?",
                "Which six-character combination opens the safe?",
                "What passphrase grants entry to the strongbox?",
            ]
        },
        {
            "needle": "Storage compartment PIN is: {CODE}",
            "questions": [
                "What numeric sequence opens the locker?",
                "Which code unlocks the deposit box?",
                "What combination secures the cabinet?",
            ]
        },
    ],
    "portal": [
        {
            "needle": "Gateway activation sequence: {CODE}",
            "questions": [
                "What string initiates the doorway?",
                "Which characters enable passage?",
                "What key opens the entryway?",
            ]
        },
        {
            "needle": "Entry mechanism cipher: {CODE}",
            "questions": [
                "What characters unlock the doorway?",
                "Which combination enables passage through?",
                "What sequence permits crossing?",
            ]
        },
    ],
    "finance": [
        {
            "needle": "Wire confirmation number: {CODE}",
            "questions": [
                "What reference validates the monetary movement?",
                "Which string confirms the fund relocation?",
                "What sequence verifies the capital shift?",
            ]
        },
        {
            "needle": "Ledger entry marker: {CODE}",
            "questions": [
                "What tag identifies the bookkeeping record?",
                "Which label marks the accounting notation?",
                "What stamp distinguishes the fiscal entry?",
            ]
        },
    ],
    "clinical": [
        {
            "needle": "Chart annotation stamp: {CODE}",
            "questions": [
                "What marker identifies the patient file?",
                "Which tag labels the health documentation?",
                "What imprint distinguishes the medical record?",
            ]
        },
        {
            "needle": "Rx validation mark: {CODE}",
            "questions": [
                "What stamp confirms the drug order?",
                "Which imprint validates the medication script?",
                "What label verifies the pharmaceutical directive?",
            ]
        },
    ],
    "contract": [
        {
            "needle": "Agreement seal imprint: {CODE}",
            "questions": [
                "What mark validates the legal arrangement?",
                "Which stamp confirms the binding document?",
                "What imprint authenticates the formal pact?",
            ]
        },
        {
            "needle": "Filing cabinet tag: {CODE}",
            "questions": [
                "What label marks the document drawer?",
                "Which identifier distinguishes the paperwork storage?",
                "What notation indexes the record repository?",
            ]
        },
    ],
    "server": [
        {
            "needle": "Backend handshake phrase: {CODE}",
            "questions": [
                "What string initiates the machine dialogue?",
                "Which sequence starts the computational exchange?",
                "What characters begin the digital conversation?",
            ]
        },
        {
            "needle": "Repository checkout hash: {CODE}",
            "questions": [
                "What fingerprint identifies the codebase snapshot?",
                "Which digest marks the software version?",
                "What signature distinguishes the source revision?",
            ]
        },
    ],
}

# ============================================================================
# 4. 규칙 기반 생성 (50개)
# ============================================================================

def generate_rule_based_samples(num_samples: int = 50) -> List[Dict]:
    """규칙 기반 NoLiMa 샘플 생성"""
    samples = []
    domains = list(NOLIMA_TEMPLATES.keys())

    for i in range(num_samples):
        # 도메인 선택
        domain = domains[i % len(domains)]
        templates = NOLIMA_TEMPLATES[domain]
        template = random.choice(templates)

        # 코드 생성
        code = generate_code()

        # Needle 생성
        needle_phrase = template['needle'].format(CODE=code)

        # 질문 선택 (다양성)
        question = random.choice(template['questions'])

        # Haystack 구성 (도메인별 단락 조합)
        haystack_domain = random.choice(list(HAYSTACK_PARAGRAPHS.keys()))
        paragraphs = random.sample(HAYSTACK_PARAGRAPHS[haystack_domain], min(3, len(HAYSTACK_PARAGRAPHS[haystack_domain])))

        # 다른 도메인에서 추가
        other_domains = [d for d in HAYSTACK_PARAGRAPHS.keys() if d != haystack_domain]
        for other in random.sample(other_domains, min(2, len(other_domains))):
            paragraphs.append(random.choice(HAYSTACK_PARAGRAPHS[other]))

        random.shuffle(paragraphs)

        # Needle 삽입 위치 (25%-75% 사이)
        insertion_point = random.randint(1, len(paragraphs) - 1)
        paragraphs.insert(insertion_point, needle_phrase)

        context = " ".join(paragraphs)

        sample = {
            "context": context,
            "needle_phrase": needle_phrase,
            "question": question,
            "answer": code,
            "domain": domain,
            "generation_method": "rule_based",
            "context_length_chars": len(context),
        }

        # 검증
        valid, reason = validate_sample(sample)
        if valid:
            samples.append(sample)
        else:
            print(f"  Warning: Sample {i} failed validation: {reason}")
            # 재시도
            i -= 1

    return samples

# ============================================================================
# 5. Gemini 기반 생성 (150개)
# ============================================================================

def generate_gemini_sample(domain: str) -> Dict:
    """Gemini CLI로 단일 샘플 생성"""
    code = generate_code()

    prompt = f'''Generate a NoLiMa-style passage for domain: {domain}

STRICT REQUIREMENTS:
1. Write a coherent 200-300 word paragraph about {domain}
2. Naturally embed this EXACT phrase somewhere in the middle:
   "The {domain} system uses identifier: {code}"
3. DO NOT use words like "identifier", "system", "uses" in any other sentence
4. The text must be professional and realistic

OUTPUT FORMAT (JSON only, no explanation):
{{"context": "your paragraph here with the identifier embedded", "needle_phrase": "The {domain} system uses identifier: {code}"}}'''

    try:
        result = subprocess.run(
            ['gemini', '-m', 'gemini-2.0-flash', '-o', 'text', prompt],
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout.strip()

        # JSON 파싱 시도
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())

            # 질문 생성 (Gemini에게 별도 요청)
            q_prompt = f'''Create a question asking about "{code}" WITHOUT using words: identifier, system, uses, {domain}
Example: "What credential is required for access?"
Reply with ONLY the question, nothing else.'''

            q_result = subprocess.run(
                ['gemini', '-m', 'gemini-2.0-flash', '-o', 'text', q_prompt],
                capture_output=True,
                text=True,
                timeout=15
            )

            question = q_result.stdout.strip()
            if not question.endswith('?'):
                question = "What authorization credential is mentioned in the text?"

            return {
                "context": data.get('context', ''),
                "needle_phrase": data.get('needle_phrase', f"The {domain} system uses identifier: {code}"),
                "question": question,
                "answer": code,
                "domain": domain,
                "generation_method": "gemini",
                "context_length_chars": len(data.get('context', '')),
            }

    except Exception as e:
        print(f"  Gemini error: {e}")

    return None

def generate_gemini_samples(num_samples: int = 150) -> List[Dict]:
    """Gemini CLI로 다수 샘플 생성"""
    samples = []
    domains = ["technology", "business", "healthcare", "legal", "finance"]

    print(f"  Generating {num_samples} samples via Gemini CLI...")

    for i in range(num_samples):
        domain = domains[i % len(domains)]
        print(f"  [{i+1}/{num_samples}] Domain: {domain}", end=" ")

        sample = generate_gemini_sample(domain)

        if sample:
            valid, reason = validate_sample(sample)
            if valid:
                samples.append(sample)
                print("✓")
            else:
                print(f"✗ ({reason})")
                # 폴백: 규칙 기반으로 대체
                fallback = generate_rule_based_samples(1)
                if fallback:
                    fallback[0]['generation_method'] = 'fallback'
                    samples.append(fallback[0])
                    print(f"    → Fallback used")
        else:
            print("✗ (Gemini failed)")
            # 폴백
            fallback = generate_rule_based_samples(1)
            if fallback:
                fallback[0]['generation_method'] = 'fallback'
                samples.append(fallback[0])

    return samples

# ============================================================================
# 6. 메인 함수
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate NoLiMa-style NIAH dataset")
    parser.add_argument("--rule-based", type=int, default=50, help="Number of rule-based samples")
    parser.add_argument("--gemini", type=int, default=150, help="Number of Gemini-generated samples")
    parser.add_argument("--output", type=str, default="data/nolima/nolima_200.jsonl")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini generation (rule-based only)")
    args = parser.parse_args()

    print("=" * 70)
    print("NoLiMa 스타일 NIAH 데이터셋 생성")
    print("=" * 70)

    all_samples = []

    # 1. 규칙 기반 생성
    print(f"\n📝 Phase 1: 규칙 기반 생성 ({args.rule_based}개)")
    rule_samples = generate_rule_based_samples(args.rule_based)
    all_samples.extend(rule_samples)
    print(f"  ✓ {len(rule_samples)}개 생성 완료")

    # 2. Gemini 생성 (선택적)
    if not args.skip_gemini and args.gemini > 0:
        print(f"\n🤖 Phase 2: Gemini 생성 ({args.gemini}개)")
        gemini_samples = generate_gemini_samples(args.gemini)
        all_samples.extend(gemini_samples)
        print(f"  ✓ {len(gemini_samples)}개 생성 완료")

    # 3. 셔플
    random.shuffle(all_samples)

    # 4. 통계
    print(f"\n📊 최종 통계")
    print(f"  총 샘플 수: {len(all_samples)}")

    overlaps = [calculate_lexical_overlap(s['question'], s['needle_phrase']) for s in all_samples]
    ttrs = [calculate_ttr(s['context']) for s in all_samples]

    print(f"  평균 Lexical Overlap: {sum(overlaps)/len(overlaps)*100:.2f}%")
    print(f"  평균 TTR: {sum(ttrs)/len(ttrs):.3f}")

    method_counts = {}
    for s in all_samples:
        method = s.get('generation_method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
    print(f"  생성 방법: {method_counts}")

    # 5. 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n💾 저장 완료: {output_path}")
    print(f"=" * 70)

if __name__ == "__main__":
    main()
