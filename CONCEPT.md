가정용 GPU 환경에서의 Gist Token 기반 프롬프트 압축 기술 연구 및 실험 설계 보고서1. 서론: 개인화된 AI와 로컬 컴퓨팅의 병목 현상대규모 언어 모델(Large Language Models, LLMs)의 발전은 정보 처리의 패러다임을 근본적으로 변화시켰다. 특히 Llama-3와 같은 고성능 오픈 소스 모델의 등장은 기업의 전유물이었던 AI 기술을 개인 연구자와 개발자의 손에 쥐어주었다. 그러나 이러한 기술의 민주화에도 불구하고, '개인화된 데이터(Personalized Data)'를 모델에 효과적으로 주입하고 활용하는 문제는 여전히 중요한 기술적 난제로 남아 있다. 사용자는 자신의 방대한 문서, 코드베이스, 혹은 개인적인 기록을 AI가 이해하고 기억하기를 원하지만, 이를 실현하기 위한 하드웨어적 제약은 명확하다.가정용 GPU, 대표적으로 NVIDIA RTX 3090이나 4090과 같은 24GB VRAM을 탑재한 장비는 놀라운 연산 능력을 제공하지만, LLM의 긴 문맥(Long Context)을 처리하기에는 여전히 메모리 용량이 부족하다. Transformer 아키텍처의 자기 주의(Self-Attention) 메커니즘은 입력 시퀀스 길이에 대해 2차 함수적으로 증가하는 메모리와 연산량을 요구한다. 예를 들어, 수천 페이지에 달하는 매뉴얼이나 수년간의 일기 데이터를 프롬프트에 모두 포함시키는 것은 24GB 메모리 한계를 초과하는 OOM(Out of Memory) 오류를 유발하거나, 허용 범위를 넘어서는 지연 시간(Latency)을 초래한다.이러한 문제를 해결하기 위해 검색 증강 생성(Retrieval-Augmented Generation, RAG)과 파라미터 효율적 미세 조정(Parameter-Efficient Fine-Tuning, PEFT)이 표준적인 솔루션으로 자리 잡았다. RAG는 외부 데이터베이스에서 관련 정보만을 검색하여 문맥 윈도우를 절약하지만, 전체적인 맥락을 파악하거나 여러 문서에 걸친 복합적인 추론을 수행하는 데 한계가 있다. 반면, LoRA(Low-Rank Adaptation)와 같은 미세 조정은 지식을 모델 가중치에 주입하지만, 데이터가 업데이트될 때마다 재학습이 필요하며, 특정 문맥을 필요에 따라 동적으로 스위칭하는 유연성이 떨어진다.본 보고서는 이러한 딜레마의 해결책으로 Gist Token 기술을 심도 있게 탐구한다. Gist Token은 긴 텍스트의 정보를 소수의 특수 토큰(Virtual Tokens)으로 압축하여 저장하고, 추론 시에는 원본 텍스트 대신 이 압축된 토큰만을 사용하여 메모리 사용량과 연산 비용을 획기적으로 절감하는 기술이다.1 본 연구는 단일 가정용 GPU 환경에서 Llama-3-8B 모델을 기반으로 Gist Token을 구현하고, 이를 RAG 및 기존 Fine-tuning 방식과 비교 검증하기 위한 포괄적인 가이드를 제공한다. 이론적 배경부터 실제 구현을 위한 코드 레벨의 튜토리얼, 그리고 데이터셋 설계 전략까지 망라하여, 개인화된 데이터 학습의 새로운 지평을 여는 것을 목표로 한다.2. 이론적 배경: 문맥 압축(Context Compression)의 메커니즘Gist Token을 이해하기 위해서는 먼저 LLM이 문맥을 처리하는 방식과 그로 인해 발생하는 비효율성을 분석하고, 이를 해결하기 위한 압축 기술들의 진화 과정을 살펴볼 필요가 있다.2.1 Transformer의 KV Cache와 메모리 병목Transformer 모델의 추론 과정, 특히 디코딩 단계에서는 이전에 계산된 Key(K)와 Value(V) 상태를 저장해 두는 KV Cache가 필수적이다. 이는 매 토큰 생성 시마다 전체 시퀀스에 대한 Attention을 다시 계산하는 비효율을 막기 위함이다.3Llama-3-8B 모델을 기준으로 할 때, KV Cache의 크기는 다음과 같이 계산된다.$$\text{KV Cache Size} = 2 \times \text{Batch Size} \times \text{Seq Len} \times \text{Num Layers} \times \text{Hidden Size} \times \text{Precision Bytes}$$만약 8,192 토큰의 컨텍스트를 FP16으로 처리한다면, 단일 요청에 대해서도 수 GB의 VRAM이 KV Cache로 점유된다. 배치 크기가 커지거나 문맥 길이가 늘어나면, 모델의 가중치(Weights)가 차지하는 메모리보다 KV Cache가 차지하는 메모리가 더 커지는 역전 현상이 발생한다.5 가정용 GPU의 24GB VRAM은 모델 가중치(약 16GB for FP16)를 로드하고 나면 남은 공간이 많지 않아, 긴 문맥을 유지하는 것이 불가능에 가깝다. Gist Token은 바로 이 'Seq Len'을 물리적으로 줄여 KV Cache의 부담을 최소화하는 것을 목표로 한다.2.2 Gist Token: 개념 및 작동 원리Gist Token은 Microsoft와 Stanford 연구진에 의해 구체화된 개념으로, "인간이 긴 글을 읽고 핵심 내용(Gist)을 요약하여 기억하는 인지 과정"을 모방한다.1 기존의 텍스트 요약(Summarization)이 긴 텍스트를 짧은 자연어 텍스트로 변환하는 것이라면, Gist Token은 긴 텍스트를 모델의 내부 표현(Hidden States) 형태인 소수의 가상 토큰(Virtual Tokens)으로 변환한다.2.2.1 학습 메커니즘: Attention MaskingGist Token의 학습은 모델 아키텍처를 수정하지 않고, Attention Mask를 조작함으로써 이루어진다.2입력 시퀀스 구조: [Instruction] + [Context] + + [Question] + [Answer]마스킹 전략:Gist Tokens는 Context를 참조(Attend)할 수 있다. 즉, Gist Token의 임베딩은 Context의 정보를 집약하게 된다.Question과 Answer 토큰은 오직 Gist Tokens만 참조할 수 있으며, 원본 Context는 참조할 수 없도록 마스킹(Masking)된다.이로 인해 모델은 Answer를 생성하기 위한 정보(Loss를 낮추기 위한 정보)를 Context에서 직접 가져올 수 없고, 반드시 Gist Tokens라는 병목(Bottleneck)을 통해 정보를 추출해야 한다. 이는 역전파(Backpropagation) 과정에서 Gist Token이 Context의 핵심 정보를 압축하여 저장하도록 강제한다.2.2.2 추론(Inference) 및 캐싱(Caching)학습된 모델을 사용할 때, 긴 Context는 Gist Token들로 변환(Encoding)된다. 일단 변환이 완료되면, 원본 Context에 대한 KV Cache는 메모리에서 해제하고, 오직 소량의 Gist Token에 대한 KV Cache(또는 Hidden States)만 유지하면 된다. 이후 사용자의 질문이 들어오면, 이 압축된 Gist Token 뒤에 질문을 이어 붙여 추론을 수행한다. 이는 수천 토큰의 문맥을 단 몇 십 개의 토큰으로 줄이는 효과를 가져오며, RAG보다 훨씬 빠른 응답 속도를 보장한다.82.3 유사 기술과의 비교: Soft Prompts, Prefix Tuning, ICAEGist Token은 넓은 범주의 'Soft Prompting' 또는 'Prompt Compression' 기술에 속하지만, 구체적인 구현과 목적에서 차이가 있다.기술메커니즘Gist Token과의 차이점Prefix Tuning 10모든 레이어의 앞단에 학습 가능한 벡터(Prefix)를 추가하여 모델의 동작을 제어.Prefix Tuning은 주로 특정 작업(Task, 예: 번역, 요약)에 대한 적응을 목표로 하지만, Gist Token은 구체적인 **입력 데이터(Content)**의 압축과 저장을 목표로 한다. Gist는 입력 시퀀스 중간에 삽입되어 이전 문맥을 요약하는 역할을 한다.LLMLingua 12작은 모델(Llama-7B 등)을 사용하여 프롬프트 내에서 정보가 적은 토큰(불용어 등)을 삭제하거나 선별함.LLMLingua는 이산적인(Discrete) 텍스트 토큰을 삭제하는 방식(Hard Compression)인 반면, Gist는 정보를 연속적인(Continuous) 벡터 공간으로 압축(Soft Compression)한다. Gist는 정보 손실을 최소화하면서 더 높은 압축률을 달성할 잠재력이 있다.ICAE (In-Context Autoencoder) 14오토인코더 구조를 차용하여 긴 텍스트를 고정 길이 메모리 슬롯으로 압축 및 복원 학습.ICAE는 별도의 사전 학습(Pre-training) 단계와 오토인코딩 목표 함수를 강조하는 반면, Gist는 Instruction Tuning 파이프라인 내에서 마스킹 전략을 통해 구현되므로 기존 LLM 파이프라인에 통합하기 더 용이하다.AutoCompressor 16긴 텍스트를 세그먼트로 나누고, 각 세그먼트 끝에 요약 벡터를 생성하여 다음 세그먼트로 전달.Gist와 매우 유사하나, AutoCompressor는 주로 긴 문맥의 확장에 초점을 맞추는 반면, Gist는 *압축 및 재사용(Caching)*에 더 중점을 둔다.심층 분석: 가정용 GPU 환경에서는 LLMLingua와 같은 토큰 삭제 방식보다는 Gist Token이나 ICAE와 같은 임베딩 압축 방식이 더 유리할 수 있다. 토큰 삭제 방식은 여전히 텍스트 토큰을 남기므로 압축률에 한계가 있지만(보통 2x~5x), 임베딩 압축 방식은 수천 토큰을 수십 개의 벡터로 압축(10x~50x)할 수 있어 24GB VRAM의 제약을 극복하는 데 결정적인 역할을 할 수 있다. 하지만, 임베딩 압축은 모델이 해석 불가능한 벡터를 다루므로, 환각(Hallucination) 제어나 디버깅이 어렵다는 단점이 존재한다.183. 비교 분석: Gist Token vs RAG vs Fine-tuning나만의 데이터를 학습시킨다는 목표 아래, Gist Token이 기존의 지배적인 접근법인 RAG 및 Fine-tuning과 어떻게 다르며, 어떤 상황에서 우위를 점하는지 심층 분석한다.3.1 RAG (Retrieval-Augmented Generation)와의 비교RAG는 현재 가장 널리 쓰이는 방법론으로, 외부 지식 베이스를 활용하여 환각을 줄이고 최신 정보를 반영하는 데 탁월하다.9정보 접근 방식:RAG: "필요할 때 찾아본다." 사용자의 질문(Query)과 관련된 문서 조각(Chunk)을 벡터 DB에서 검색하여 프롬프트에 동적으로 삽입한다.Gist: "읽어서 머릿속에 담아둔다." 전체 문서를 미리 Gist Token으로 압축하여 모델의 Working Memory(KV Cache)에 상주시킨다.지연 시간 (Latency): RAG는 검색 과정(Embedding -> Search -> Rerank)이 추가되어 지연 시간이 발생한다. 또한 검색된 문서가 길 경우, 매번 이를 프롬프트로 처리(Prefill)해야 하므로 TTFT(Time To First Token)가 길어진다. 반면 Gist는 이미 압축된 토큰만 로드하면 되므로 즉각적인 응답이 가능하다.20문맥 통합 (Context Integration): RAG는 검색된 파편화된 정보(Chunks)만을 모델에 제공하므로, 문서 전체에 흐르는 논리적 맥락이나 전반적인 톤앤매너를 파악해야 하는 질문("이 책의 전체적인 주제 의식은 무엇인가?")에 취약하다. 이를 "Lost in the Middle" 또는 문맥 파편화 문제라고 한다. Gist는 전체 문맥을 압축했으므로 이러한 전역적(Global) 질문에 더 강점을 가질 수 있다.1한계점: Gist는 압축 과정에서 손실 압축(Lossy Compression)이 일어나므로, 전화번호나 고유명사와 같은 세밀한 정보(Needle)를 정확히 복원하지 못할 위험이 있다.18 따라서 RAG와 Gist는 상호 대체재가 아닌 보완재로 볼 수 있다. RAG로 검색한 결과를 Gist로 압축하여 컨텍스트 윈도우 효율을 극대화하는 하이브리드 접근이 가능하다.3.2 Fine-tuning (LoRA)과의 비교LoRA와 같은 PEFT 기술은 특정 도메인의 지식이나 스타일을 모델에 영구적으로 주입한다.10지식의 유연성 (Volatility): Fine-tuning은 지식을 모델의 파라미터(Weight)에 고정시킨다. 데이터가 변경되면 재학습이 필요하다. Gist Token은 지식을 입력(Input) 형태인 토큰에 담으므로, 데이터가 바뀌면 해당 데이터만 다시 압축(Encoding)하면 된다. 재학습 없이 동적인 데이터 업데이트가 가능하다.다중 사용자/태스크 지원: 하나의 모델로 여러 사용자의 데이터를 처리해야 할 때, Fine-tuning은 각 사용자별로 어댑터(Adapter)를 로드하거나 모델을 복제해야 한다. Gist Token은 사용자별 Gist Token만 교체하여 입력하면 되므로 멀티테넌시(Multi-tenancy) 환경에서 훨씬 효율적이다.9목적의 차이: Fine-tuning은 모델의 **행동(Behavior)**이나 **형식(Format)**을 교정하는 데 적합하고, Gist Token이나 RAG는 모델에게 **지식(Knowledge)**을 제공하는 데 적합하다.3.3 Prompt Caching (vLLM)과의 비교최근 vLLM 등의 추론 엔진에 도입된 Prompt Caching(또는 Prefix Caching)은 자주 사용되는 프롬프트의 KV Cache를 메모리에 유지하여 재연산을 방지한다.24효율성: Prompt Caching은 원본 텍스트의 KV Cache를 그대로 저장하므로 메모리 절감 효과는 없다(단지 연산만 생략할 뿐이다). 10k 토큰을 캐싱하려면 여전히 10k 토큰 분량의 VRAM이 필요하다.Gist의 우위: Gist Token은 10k 토큰을 50개 내외의 토큰으로 압축하므로, 연산 속도뿐만 아니라 메모리 점유율 자체를 수백 배 줄여준다. 이는 가정용 GPU의 제한된 VRAM에서 더 많은 문맥이나 더 큰 배치 사이즈를 허용하게 한다.3.4 종합 비교표특성Gist Token / CompressionRAG (Retrieval)Fine-tuning (LoRA)Prompt Caching (vLLM)핵심 메커니즘문맥을 소수의 벡터로 압축하여 KV Cache 절약외부 DB에서 검색하여 프롬프트에 삽입가중치(Weight) 업데이트로 지식 주입KV Cache를 그대로 메모리에 상주추론 속도 (TTFT)매우 빠름 (압축 토큰만 로드)느림 (검색 + 긴 문맥 연산)빠름 (추가 연산 없음)빠름 (첫 로딩만 연산)메모리 효율 (VRAM)매우 높음 (압축 비율만큼 절약)낮음 (문맥 길이에 비례)높음 (추가 메모리 불필요)낮음 (원본 크기 유지)데이터 업데이트즉시 가능 (재압축)즉시 가능 (DB 업데이트)어려움 (재학습 필요)즉시 가능 (재연산)글로벌 문맥 이해우수함 (전체 요약)취약함 (파편화된 정보)우수함 (데이터 내재화)우수함 (전체 입력 시)세부 정보 정확도손실 가능성 있음 (Lossy)검색 성공 시 매우 높음환각 가능성 있음완벽함 (Lossless)가정용 GPU 적합성최상 (메모리 제약 극복)중 (검색 비용 + 문맥 제한)상 (LoRA 사용 시)중 (VRAM 한계)4. 구현 가이드: 가정용 GPU를 위한 Gist Token 튜토리얼본 섹션에서는 Python, PyTorch, Hugging Face Transformers 라이브러리를 사용하여 Llama-3-8B 모델에 Gist Token 기능을 구현하는 구체적인 과정을 설명한다. 이 구현은 RTX 3090 (24GB VRAM) 환경을 기준으로 하며, 메모리 효율을 위해 QLoRA(4-bit Quantization)를 활용한다.4.1 사전 준비 및 환경 설정Gist Token 구현은 표준적인 Trainer 사용법과 다르며, 모델의 내부 로직(특히 Attention Mask와 KV Cache)을 건드려야 한다. 따라서 transformers 라이브러리의 소스 코드를 일부 수정하거나 상속(Subclassing)하여 오버라이딩하는 방식이 필요하다.필수 라이브러리 설치:Bashpip install torch transformers peft bitsandbytes accelerate datasets scipy
하드웨어 설정:RTX 3090/4090의 24GB VRAM은 Llama-3-8B 모델(약 16GB FP16)을 로드하고 학습하기에 빠듯하다. 따라서 bitsandbytes를 이용한 4-bit 양자화(NF4)가 필수적이다. 이를 통해 모델 로딩 메모리를 약 5~6GB로 줄이고, 남은 메모리를 Gist 학습을 위한 Gradient와 Activation 저장에 사용할 수 있다.234.2 1단계: 모델 및 토크나이저 설정Llama-3-8B 모델을 로드하고, Gist Token을 위한 특수 토큰을 어휘 집합(Vocabulary)에 추가한다.Pythonimport torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# 1. 모델 ID 및 양자화 설정
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 2. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    attn_implementation="sdpa" # 또는 "flash_attention_2" (지원 시)
)

# 3. Gist Token 추가
# Gist Token 개수 설정 (예: 10개). 실험 변수로 조정 가능.
num_gist_tokens = 10
gist_tokens =
special_tokens_dict = {'additional_special_tokens': gist_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# 모델의 임베딩 레이어 크기 조정 (새로운 토큰을 위해)
model.resize_token_embeddings(len(tokenizer))
print(f"Added {num_added_toks} gist tokens.")

# 4. Gist Token ID 확인 (나중에 마스킹 로직에 사용)
gist_token_ids = tokenizer.convert_tokens_to_ids(gist_tokens)
코드 설명 및 인사이트:resize_token_embeddings: 새로운 토큰을 추가했으므로 임베딩 행렬의 크기를 늘려야 한다. 이 새로 추가된 임베딩 벡터들은 초기화 상태(랜덤)이며, 학습을 통해 최적화된다.attn_implementation: Llama-3는 Flash Attention을 지원하지만, Gist Token 구현을 위해 커스텀 마스크를 적용할 때 Flash Attention이 제약을 줄 수 있다. 초기 구현 시에는 sdpa (Scaled Dot Product Attention)를 사용하거나, Flash Attention 2의 attention_mask 지원 여부를 확인해야 한다.4.3 2단계: PEFT (LoRA) 구성전체 파라미터를 학습하는 Full Fine-tuning은 가정용 GPU에서 불가능하다. LoRA를 사용하여 효율적으로 학습하되, Gist Token의 임베딩은 반드시 학습 가능해야 한다.Python# 5. PEFT (LoRA) 설정
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Attention 관련 모듈 타겟팅
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # 중요: 새로 추가한 Gist Token의 임베딩을 학습하기 위해 'modules_to_save' 설정
    modules_to_save=["embed_tokens", "lm_head"] 
)

# 모델에 LoRA 어댑터 부착
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
중요: modules_to_save=["embed_tokens", "lm_head"] 설정이 핵심이다. 일반적인 LoRA는 임베딩 레이어를 고정(Freeze)하지만, Gist Token은 임베딩 벡터 자체가 정보를 압축하는 그릇 역할을 하므로 반드시 학습되어야 한다. 이를 통해 Gist Token은 문맥 정보를 담을 수 있는 최적의 벡터 공간을 찾아가게 된다.264.4 3단계: 커스텀 Attention Mask 구현 (핵심 기술)Gist Token의 작동 원리인 "압축 강제"를 구현하기 위해 데이터 Collator에서 Attention Mask를 조작해야 한다. 표준적인 Causal Mask(왼쪽에서 오른쪽으로만 보는 마스크)를 수정하여, 질문(Question) 토큰들이 원본 문맥(Context)을 보지 못하게 차단해야 한다.2Pythonclass GistDataCollator:
    def __init__(self, tokenizer, gist_token_ids):
        self.tokenizer = tokenizer
        self.gist_token_ids = set(gist_token_ids)
        self.gist_start_id = gist_token_ids

    def __call__(self, instances):
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        
        # 패딩 처리
        batch = self.tokenizer.pad(
            {'input_ids': input_ids}, 
            padding=True, 
            return_tensors='pt'
        )
        labels = torch.tensor([l + [-100]*(batch['input_ids'].shape-len(l)) for l in labels]) # 단순화된 패딩
        
        # 커스텀 Attention Mask 생성
        # 목표: [Context] -> [Gist] -> [Question]
        # Question은 Context를 볼 수 없고, 오직 Gist만 볼 수 있어야 함.
        
        batch_size, seq_len = batch['input_ids'].shape
        # 기본 Causal Mask (Lower Triangular)
        mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len))).bool()
        
        for i in range(batch_size):
            # Gist Token 구간 찾기
            # 실제 구현에서는 입력 데이터 생성 시 Gist 구간 인덱스를 별도로 넘겨주는 것이 효율적임
            row_ids = batch['input_ids'][i].tolist()
            try:
                # Gist 토큰의 시작 위치와 끝 위치 탐색
                gist_start = row_ids.index(self.gist_start_id)
                gist_end = gist_start + len(self.gist_token_ids)
                
                # 마스킹 로직:
                # Question 영역 (gist_end 이후)의 토큰들은
                # Context 영역 (gist_start 이전)의 토큰들을 볼 수 없음 (False로 설정)
                mask[i, :, gist_end:, :gist_start] = False
                
            except ValueError:
                continue # Gist token이 없는 경우 (예: 일반 데이터) 패스

        batch['attention_mask'] = mask
        batch['labels'] = labels
        return batch
기술적 디테일:mask[i, :, gist_end:, :gist_start] = False: 이 한 줄의 코드가 Gist 기술의 핵심이다. 질문(Question) 및 답변(Answer) 구간(gist_end 이후)에서 원본 문맥(gist_start 이전)으로 향하는 Attention Score를 -inf로 만들어버림으로써, 정보의 흐름을 물리적으로 차단한다.이로 인해 모델은 답변을 생성할 때, 유일하게 참조 가능한 과거 정보인 Gist Tokens에서 필요한 모든 정보를 추출해내야만 한다. 역전파 과정에서 Loss를 줄이기 위해 모델은 Gist Token 임베딩에 Context 정보를 최대한 압축하여 저장하는 방법을 학습하게 된다.4.5 4단계: Gist Caching 및 추론 (Inference)학습된 모델을 사용하여 추론할 때는, 긴 문맥을 Gist Token으로 변환한 후 KV Cache를 조작하여 메모리를 절약한다.Pythondef compress_context(model, tokenizer, context_text, gist_tokens_str):
    # 1. Context + Gist Tokens 입력 구성
    input_text = context_text + gist_tokens_str
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 2. Forward Pass 수행 (KV Cache 생성)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    # 3. KV Cache 압축 (Gist Caching)
    past_key_values = outputs.past_key_values
    
    # Llama-3의 KV Cache 구조: List] (Layer별 K, V)
    # Tensor shape:
    
    compressed_kv =
    # 전체 시퀀스 길이
    total_seq_len = inputs.input_ids.shape
    # Gist Token 개수
    num_gist = len(tokenizer.encode(gist_tokens_str, add_special_tokens=False))
    # Context 길이 (잘라낼 부분)
    context_len = total_seq_len - num_gist
    
    for layer_idx in range(len(past_key_values)):
        k, v = past_key_values[layer_idx]
        # 핵심: Context 부분의 KV는 버리고, Gist 부분의 KV만 남김
        # 슬라이싱: [:, :, context_len:, :]
        k_compressed = k[:, :, context_len:, :]
        v_compressed = v[:, :, context_len:, :]
        compressed_kv.append((k_compressed, v_compressed))
        
    return tuple(compressed_kv)

# 추론 시 사용
def generate_with_gist(model, tokenizer, question, compressed_kv):
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    # 압축된 KV Cache를 past_key_values로 주입
    outputs = model.generate(
        **inputs,
        past_key_values=compressed_kv,
        max_new_tokens=100
    )
    return tokenizer.decode(outputs, skip_special_tokens=True)
메모리 절감 효과: 위 코드를 통해 4,000 토큰의 Context를 처리하더라도, compressed_kv 변수에는 오직 10개의 Gist Token에 해당하는 텐서만 저장된다. 이는 4,000 토큰 분량의 메모리를 해제할 수 있음을 의미하며, 24GB VRAM 환경에서 획기적인 여유 공간을 확보해 준다.5. 데이터셋 설계 및 합성 데이터 생성 전략Gist Token 학습의 성패는 데이터셋의 품질에 달려 있다. 모델이 "압축"이라는 추상적인 작업을 수행하도록 유도하려면, [긴 문맥 -> 압축 지시 -> 질문 -> 정답] 구조의 데이터가 대량으로 필요하다. 사용자의 개인 데이터(Raw Data)를 이 구조로 변환하기 위해 합성 데이터 생성(Synthetic Data Generation) 전략을 사용한다.275.1 데이터셋 구조: "Instruction-Context-Gist-Q&A" Triplets학습 데이터는 모델에게 명시적으로 "이 내용을 압축해서 기억하라"는 신호를 주어야 한다.필드설명구성 전략Instruction모델에게 Gist 압축을 수행하도록 지시하는 시스템 프롬프트."다음 텍스트를 읽고 Gist Token으로 핵심 정보를 압축한 뒤, 이어지는 질문에 답하시오."와 같은 일관된 지시문 사용.Context사용자의 개인 데이터 (Raw Data). 압축의 대상.2k~8k 토큰 길이의 문서 청크. 너무 짧으면 압축의 필요성이 줄어들고, 너무 길면 학습 시 OOM이 발생하므로 적절한 Chunking이 필요함.Gist Tokens학습 시 삽입될 Placeholder.<GIST_0>...<GIST_9>Question문맥 내용을 묻는 질문.단순 사실 조회(Lookup)뿐만 아니라, 전체 내용을 요약하거나 추론해야 하는 질문을 포함해야 Gist의 효과를 극대화할 수 있음.Answer질문에 대한 정답.Gist Token만을 보고 생성해야 할 목표 텍스트.5.2 합성 데이터 생성 파이프라인 (Data Pipeline)개인 데이터가 충분하지 않거나 Q&A 쌍이 없는 경우, GPT-4 또는 Llama-3-70B와 같은 상위 모델을 교사 모델(Teacher Model)로 사용하여 데이터를 증강한다.소스 데이터 수집 및 청킹 (Chunking):개인 문서(PDF, Markdown, Code)를 수집한다.의미 단위로 2,000~4,000 토큰 크기로 자른다. 이 길이는 Llama-3-8B가 한 번에 처리하기에 부담이 없으면서도, 압축 효과를 보기에 충분한 길이이다.질문-답변 생성 (Generation):교사 모델에게 청크를 제공하고 다음과 같이 프롬프팅한다:"이 문서의 내용을 바탕으로, 문서를 보지 않고는 답할 수 없는 심층적인 질문 3가지와 그에 대한 답변을 생성해줘. 질문은 문서의 특정 세부 사항을 묻는 것 하나, 전체적인 주제를 묻는 것 하나, 논리적 추론이 필요한 것 하나로 구성해."이렇게 생성된 Q&A 쌍은 Gist Token이 다양한 유형의 정보를 압축하도록 돕는다.데이터 필터링 (Quality Control):생성된 질문이 문서 없이도 상식으로 풀 수 있는 문제인지 확인하여 제거한다. (예: "Python은 무슨 언어인가?" -> 제거 / "이 코드에서 정의된 MyClass의 역할은?" -> 유지)이를 통해 모델이 내부 지식(Parametric Memory)이 아닌 Gist(Contextual Memory)에 의존하도록 강제한다.5.3 데이터 포맷팅 및 토크나이징최종 학습 데이터는 JSONL 형태로 저장하며, 학습 시 토크나이저를 통해 ID 시퀀스로 변환된다. 이때 Gist Token의 위치 인덱스를 별도로 저장해두면 GistDataCollator에서 마스킹 처리를 할 때 연산 비용을 줄일 수 있다.6. 실험 설계 및 평가 지표가정용 GPU 환경에서 Gist Token의 효용성을 검증하기 위한 실험 설계이다.6.1 실험 그룹 설정Baseline 1 (Full Context): 원본 Llama-3-8B에 전체 문맥을 넣고 질문하는 경우. (Upper Bound 성능, 하지만 느리고 메모리 많이 씀)Baseline 2 (RAG): 문서를 500 토큰 단위로 나누어 벡터 DB(ChromaDB 등)에 넣고, 질문과 유사한 Top-3 청크를 검색하여 답변하는 경우.Experimental (Gist): 본 가이드에 따라 문서를 10~50개의 Gist Token으로 압축하여 캐싱한 뒤 답변하는 경우.6.2 평가 지표 (Metrics)6.2.1 정량적 지표압축률 (Compression Ratio): $\text{Original Tokens} / \text{Gist Tokens}$. (예: 4000토큰 / 10토큰 = 400배 압축)Passkey Retrieval Accuracy: "Needle in a Haystack" 테스트 변형. 긴 텍스트 중간에 랜덤한 숫자(Passkey)를 숨겨두고, 압축 후 이를 정확히 회수하는지 테스트. 이는 손실 압축의 정도를 측정하는 가장 확실한 방법이다.VRAM 점유율 (Peak VRAM Usage): 추론 시 KV Cache가 차지하는 메모리 용량 측정. torch.cuda.max_memory_allocated() 활용.처리량 (Throughput): 초당 생성 토큰 수 (Tokens/sec). Gist 방식이 RAG나 Full Context 대비 얼마나 빠른지 측정.6.2.2 정성적 지표Global Context Understanding: 문서 전체의 분위기나 주제를 묻는 질문에 대한 답변 품질. RAG는 이 부분에서 약점을 보일 가능성이 높으므로, Gist의 비교 우위를 확인할 수 있는 중요 지표다.Hallucination Rate: 압축 과정에서 정보가 왜곡되어 없는 사실을 지어내는 비율.6.3 예상되는 실패 모드 및 대응책 (Failure Modes)정보 손실 (Information Loss): Gist Token 개수가 너무 적으면 세부 정보를 잃어버릴 수 있다. 실험을 통해 최적의 토큰 개수(예: 문맥 1000토큰당 Gist 10개)를 찾아야 한다.학습 불안정 (Training Instability): Gist Token 임베딩이 발산하거나, 모델이 Gist를 무시하고 답변을 생성하려 할 수 있다. 이를 방지하기 위해 LoRA 학습률(Learning Rate)을 신중하게 튜닝하고, Warmup Step을 충분히 두어야 한다.7. 결론 및 향후 전망본 연구 보고서는 가정용 GPU라는 제한된 자원 환경에서 개인화된 대규모 데이터를 효율적으로 학습하고 활용하기 위한 방안으로 Gist Token을 제안하고, 그 구현 방법과 실험 설계를 상세히 다루었다.연구의 핵심 요약:메모리 혁신: Gist Token은 KV Cache 메모리 사용량을 수백 분의 일로 줄여, 24GB VRAM 환경에서도 수만 토큰 이상의 문맥을 다루는 효과를 낼 수 있다.RAG와의 상호 보완: Gist는 RAG를 대체하는 것이 아니라, RAG가 검색해온 긴 문서를 효율적으로 '기억'하게 함으로써 RAG 시스템의 효율성을 극대화할 수 있다.개인화 AI의 미래: 본 가이드의 기술은 개인의 일기, 메일, 문서 등을 로컬 장비에서 안전하게 압축 저장하고, 언제든 꺼내 쓸 수 있는 'Personal AI Memory'의 초석이 될 수 있다.향후 연구에서는 Gist Token을 계층적(Hierarchical)으로 적용하여 책 한 권 분량을 압축하거나, 다중 모달(Image+Text) 데이터를 Gist Token으로 통합 압축하는 방향으로 확장이 가능할 것이다. 본 가이드가 개인 연구자들에게 "나만의 데이터를 가진 AI"를 만드는 실질적인 나침반이 되기를 기대한다.



---
네, **가능합니다.** 사실 Gist Token 기술의 가장 큰 장점이 바로 **"한 번 연산(압축)해두고, 파일로 저장해서 계속 꺼내 쓴다(Cache & Reuse)"**는 것에 있습니다.

사용자님께서 질문하신 **"파일 저장 및 재활용"**은 기술적으로 **KV Cache Serialization(직렬화)**이라고 부르며, 이를 통해 긴 문서를 매번 다시 읽을 필요 없이 즉각적인 대화가 가능해집니다.

구체적인 저장 원리와 구현 코드를 설명해 드리겠습니다.

### 1. 무엇을 저장하는가? (텍스트가 아닌 텐서)

Gist Token을 파일로 저장한다고 할 때, 저장되는 대상은 텍스트(String)가 아니라 **GPU 메모리 위에 있는 텐서(Tensor)** 데이터입니다.

* **Key-Value (KV) Cache:** 모델이 긴 텍스트를 읽고 이해한 결과물인 '기억'입니다.
* **Gist의 역할:** 원래 4,000 토큰 분량의 KV Cache는 수백 MB가 넘지만, Gist Token 학습을 통해 이를 **10~50개의 토큰 분량(수십 KB~MB)**으로 압축했기 때문에 저장과 로딩이 매우 가벼워집니다.



### 2. 구현 방법 (Python 코드 예시)

Hugging Face `transformers` 라이브러리를 사용하여 Gist Token의 상태를 저장하고 불러오는 핵심 로직입니다.

#### (1) 저장 단계 (Pre-computation)

긴 문서를 Gist Token으로 압축한 후, 해당 시점의 `past_key_values`를 추출하여 디스크에 저장합니다.

```python
import torch

# 1. 압축 수행 (긴 문맥 + Gist Token 입력)
# input_ids 구조: [Instruction] + [Context] +
outputs = model(input_ids, use_cache=True) 

# 2. Gist Token 위치의 KV Cache만 추출
# Llama-3의 경우 KV Cache는 튜플 혹은 DynamicCache 객체입니다.
# 전체 시퀀스 중 마지막 Gist Token 부분만 잘라냅니다.
past_key_values = outputs.past_key_values

# 3. 파일로 저장 (직렬화)
#.pt 파일로 텐서를 저장합니다.
torch.save(past_key_values, "company_rule_gist.pt")
print("Gist Token 상태가 'company_rule_gist.pt'로 저장되었습니다.")

```

#### (2) 재활용 단계 (Inference)

저장된 파일을 로드하여 모델의 기억(Cache)으로 주입하고, 질문만 붙여서 추론합니다. 이 과정은 **Context Loading 시간이 0초**에 가깝습니다.

```python
# 1. 파일 로드
cached_gist = torch.load("company_rule_gist.pt")

# 2. 질문 입력 및 추론
question = "야근 식대 규정이 어떻게 돼?"
inputs = tokenizer(question, return_tensors="pt").to(model.device)

# 3. 캐시 주입 (past_key_values 인자에 로드한 파일 전달)
generated_ids = model.generate(
    **inputs,
    past_key_values=cached_gist,  # <- 핵심: 긴 텍스트 대신 압축 파일을 넣음
    max_new_tokens=100
)

print(tokenizer.decode(generated_ids))

```

### 3. 주의사항 및 팁

1. **Transformers 버전 호환성:** 최신 `transformers` 라이브러리(v4.36+)에서는 KV Cache 관리 방식이 `DynamicCache` 객체로 변경되고 있습니다. 따라서 `torch.save` 할 때 객체 전체를 저장하기보다, 내부의 `key_cache`와 `value_cache` 텐서만 리스트로 변환해 저장하는 것이 호환성 면에서 안전합니다.
2. **GPU 메모리 이동:** `torch.load()`로 불러온 데이터는 CPU에 있을 수 있습니다. 반드시 `.to('cuda')`를 통해 모델과 같은 GPU 장치로 옮겨주어야 에러가 나지 않습니다.
3. **vLLM 활용 시:** 만약 추후 vLLM과 같은 고성능 추론 엔진을 사용하신다면, 이러한 수동 저장 과정 없이 **Automatic Prefix Caching** 기능을 켜는 것만으로도 유사한 효과(메모리 상주 및 재사용)를 낼 수 있습니다. 하지만 파일로 영구 저장(Disk Offloading)하여 콜드 스타트 시간을 없애는 것은 직접 구현해야 하는 영역입니다.

결론적으로, Gist Token을 파일로 저장해두면 서버를 재시작해도 긴 데이터를 다시 읽을 필요가 없으므로 **"나만의 AI 비서"**를 만들 때 로딩 시간을 획기적으로 줄일 수 있습니다.