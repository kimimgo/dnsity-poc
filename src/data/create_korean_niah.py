"""
한국어 NIAH (Needle in Haystack) 데이터셋 생성기.

한국어 장문 맥락에서 특정 정보 검색 능력을 테스트하기 위한 평가 데이터셋을 생성합니다.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import json
import random
import string


class KoreanNIAHGenerator:
    """
    한국어 Needle in Haystack (NIAH) 평가 데이터셋 생성기.

    한국어 장문 맥락에 "바늘"(특정 정보)을 삽입하여
    압축 알고리즘이 정확한 정보를 검색할 수 있는지 테스트합니다.
    """

    # 한국어 배경 텍스트 템플릿
    BACKGROUND_TEMPLATES = [
        "회사 정책에 따르면 직원은 항상 전문적인 태도를 유지해야 합니다. "
        "근무 시간은 오전 9시부터 오후 6시까지이며, 점심시간은 1시간입니다. "
        "원격 근무는 관리자 승인 하에 주 2회 허용됩니다. "
        "연차는 최소 2주 전에 신청해야 합니다. ",

        "인공지능 분야에서 신경망은 많은 도메인에 혁명을 일으켰습니다. "
        "딥러닝 모델은 학습을 위해 상당한 컴퓨팅 리소스가 필요합니다. "
        "전이 학습은 모델이 새로운 작업에 사전 학습된 가중치를 활용할 수 있게 합니다. "
        "어텐션 메커니즘은 자연어 처리에 매우 중요한 것으로 입증되었습니다. ",

        "기후 변화는 전 세계 생태계에 중대한 도전 과제를 제기합니다. "
        "기온 상승은 전 세계의 생물 다양성과 날씨 패턴에 영향을 미칩니다. "
        "재생 에너지원은 화석 연료에 대한 지속 가능한 대안을 제공합니다. "
        "국제 협력은 환경 문제를 해결하는 데 필수적입니다. ",

        "소프트웨어 개발은 애자일과 워터폴을 포함한 다양한 방법론을 따릅니다. "
        "Git과 같은 버전 관리 시스템은 협업 코딩을 용이하게 합니다. "
        "코드 리뷰는 코드 품질을 향상시키고 팀 간 지식 공유를 촉진합니다. "
        "지속적 통합은 테스트 및 배포 프로세스를 자동화합니다. ",

        "현대 의료 시스템은 환자 결과 개선을 위해 기술을 통합합니다. "
        "전자 건강 기록은 의료 제공자 간의 정보 공유를 간소화합니다. "
        "원격 의료는 외딴 지역의 의료 상담 접근성을 확대합니다. "
        "예방 치료는 장기 의료 비용을 줄이고 인구 건강을 개선합니다. ",

        "데이터 과학은 통계학과 컴퓨터 과학의 융합 분야입니다. "
        "머신러닝 알고리즘은 대규모 데이터에서 패턴을 발견합니다. "
        "데이터 시각화는 복잡한 정보를 이해하기 쉽게 만듭니다. "
        "윤리적 데이터 사용은 개인정보 보호에 필수적입니다. ",

        "교육 시스템은 21세기 역량 개발에 중점을 두고 있습니다. "
        "온라인 학습 플랫폼은 교육 접근성을 크게 향상시켰습니다. "
        "협력 학습은 학생들의 문제 해결 능력을 키웁니다. "
        "개인화된 교육은 각 학생의 학습 속도를 존중합니다. ",

        "금융 기술은 전통적인 은행 서비스를 혁신하고 있습니다. "
        "블록체인 기술은 거래의 투명성과 보안을 강화합니다. "
        "모바일 결제는 현금 없는 사회로의 전환을 가속화합니다. "
        "자동화된 투자 서비스는 개인 투자자의 진입 장벽을 낮춥니다. ",
    ]

    def __init__(
        self,
        output_dir: str,
        num_samples: int = 100,
        context_length_range: Tuple[int, int] = (2000, 8000)
    ):
        """
        한국어 NIAH 생성기 초기화.

        Args:
            output_dir: 데이터셋 저장 디렉토리
            num_samples: 생성할 샘플 수
            context_length_range: (최소, 최대) 컨텍스트 길이 (토큰 단위)
        """
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.context_length_range = context_length_range
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_background_text(self, target_chars: int) -> str:
        """
        목표 문자 수에 도달하도록 배경 텍스트 생성.

        Args:
            target_chars: 목표 문자 수

        Returns:
            생성된 배경 텍스트
        """
        result = []
        current_length = 0

        while current_length < target_chars:
            # 랜덤 템플릿 선택
            template = random.choice(self.BACKGROUND_TEMPLATES)
            result.append(template)
            current_length += len(template)

        return "".join(result)[:target_chars]

    def generate_needle(self) -> str:
        """
        랜덤 "바늘" (비밀 정보) 생성.

        Returns:
            랜덤 6자리 알파벳+숫자 패스키
        """
        # 6자리 알파벳+숫자 패스키 생성
        chars = string.ascii_uppercase + string.digits
        passkey = ''.join(random.choices(chars, k=6))
        return passkey

    def insert_needle(
        self,
        context: str,
        needle: str,
        position: float
    ) -> str:
        """
        컨텍스트의 지정된 상대 위치에 바늘 삽입.

        Args:
            context: 배경 텍스트
            needle: 삽입할 비밀 정보
            position: 상대 위치 (0.0 = 시작, 1.0 = 끝)

        Returns:
            바늘이 삽입된 컨텍스트
        """
        # 삽입 인덱스 계산
        insert_idx = int(len(context) * position)

        # 바늘 문장 생성 (한국어)
        needle_sentence = f" 비밀 패스키는 {needle}입니다. "

        # 컨텍스트에 삽입
        result = context[:insert_idx] + needle_sentence + context[insert_idx:]

        return result

    def generate_sample(
        self,
        target_length: int,
        needle_position: float
    ) -> Dict:
        """
        단일 NIAH 샘플 생성.

        Args:
            target_length: 목표 컨텍스트 길이 (토큰 단위, 근사치)
            needle_position: 바늘 삽입 상대 위치

        Returns:
            context, question, answer, metadata가 포함된 딕셔너리
        """
        # 배경 텍스트 생성 (한국어는 2자 ~= 1토큰 근사)
        target_chars = target_length * 2
        background = self.generate_background_text(target_chars)

        # 바늘 생성
        needle = self.generate_needle()

        # 바늘 삽입
        context = self.insert_needle(background, needle, needle_position)

        # 샘플 생성
        sample = {
            "context": context,
            "question": "텍스트에서 언급된 비밀 패스키는 무엇인가요?",
            "answer": needle,
            "needle": needle,
            "needle_position": needle_position,
            "context_length_chars": len(context),
            "language": "korean"
        }

        return sample

    def generate_all(self) -> List[Dict]:
        """
        다양한 길이와 위치로 모든 샘플 생성.

        Returns:
            NIAH 샘플 리스트
        """
        samples = []

        for i in range(self.num_samples):
            # 컨텍스트 길이 변화
            min_len, max_len = self.context_length_range
            target_length = random.randint(min_len, max_len)

            # 바늘 위치 변화 (매우 처음/끝 회피)
            needle_position = random.uniform(0.2, 0.8)

            sample = self.generate_sample(target_length, needle_position)
            samples.append(sample)

        return samples

    def save_jsonl(self, samples: List[Dict], filename: str = "korean_niah.jsonl") -> Path:
        """
        샘플을 JSONL 파일로 저장.

        Args:
            samples: 저장할 샘플 리스트
            filename: 출력 파일명

        Returns:
            저장된 파일 경로
        """
        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        return output_file


def main():
    """커맨드라인 사용을 위한 메인 진입점."""
    import argparse

    parser = argparse.ArgumentParser(description="한국어 NIAH 평가 데이터셋 생성")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="생성할 샘플 수"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/niah",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2000,
        help="최소 컨텍스트 길이 (토큰)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="최대 컨텍스트 길이 (토큰)"
    )
    parser.add_argument(
        "--output",
        default="korean_niah.jsonl",
        help="출력 파일명"
    )

    args = parser.parse_args()

    # 생성기 생성
    generator = KoreanNIAHGenerator(
        output_dir=args.output_dir,
        num_samples=args.samples,
        context_length_range=(args.min_length, args.max_length)
    )

    print(f"한국어 NIAH 샘플 {args.samples}개 생성 중...")
    samples = generator.generate_all()

    print(f"{args.output_dir}/{args.output}에 저장 중...")
    output_file = generator.save_jsonl(samples, args.output)

    print(f"✅ {len(samples)}개 샘플 생성 완료")
    print(f"📁 저장 위치: {output_file}")

    # 통계 출력
    avg_length = sum(s["context_length_chars"] for s in samples) / len(samples)
    print(f"📊 평균 컨텍스트 길이: {avg_length:.0f}자 (~{avg_length/2:.0f} 토큰)")


if __name__ == "__main__":
    main()
