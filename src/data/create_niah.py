"""
NIAH (Needle in Haystack) dataset generator.

Generates synthetic evaluation datasets for testing context compression algorithms'
ability to retain specific information embedded in long contexts.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import random
import string


class NIAHGenerator:
    """
    Generator for Needle in Haystack (NIAH) evaluation datasets.

    Creates synthetic long contexts with embedded "needles" (specific information)
    to test whether compression algorithms can retrieve exact information.
    """

    # Template texts for background generation
    BACKGROUND_TEMPLATES = [
        "The company policy states that employees must adhere to professional standards at all times. "
        "Working hours are from 9 AM to 6 PM, with a one-hour lunch break. "
        "Remote work is permitted twice a week with manager approval. "
        "Annual leave must be requested at least two weeks in advance. ",

        "In the field of artificial intelligence, neural networks have revolutionized many domains. "
        "Deep learning models require substantial computational resources for training. "
        "Transfer learning allows models to leverage pre-trained weights for new tasks. "
        "Attention mechanisms have proven crucial for natural language processing. ",

        "Climate change poses significant challenges to global ecosystems. "
        "Rising temperatures affect biodiversity and weather patterns worldwide. "
        "Renewable energy sources offer sustainable alternatives to fossil fuels. "
        "International cooperation is essential for addressing environmental issues. ",

        "Software development follows various methodologies including Agile and Waterfall. "
        "Version control systems like Git facilitate collaborative coding. "
        "Code reviews improve code quality and knowledge sharing among teams. "
        "Continuous integration automates testing and deployment processes. ",

        "Modern healthcare systems integrate technology for improved patient outcomes. "
        "Electronic health records streamline information sharing between providers. "
        "Telemedicine expands access to medical consultations in remote areas. "
        "Preventive care reduces long-term healthcare costs and improves population health. "
    ]

    def __init__(
        self,
        output_dir: str,
        num_samples: int = 100,
        context_length_range: Tuple[int, int] = (2000, 8000)
    ):
        """
        Initialize NIAH generator.

        Args:
            output_dir: Directory to save generated datasets
            num_samples: Number of samples to generate
            context_length_range: (min, max) context length in tokens
        """
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.context_length_range = context_length_range
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_background_text(self, target_chars: int) -> str:
        """
        Generate background text to reach target character count.

        Args:
            target_chars: Target number of characters

        Returns:
            Generated background text
        """
        result = []
        current_length = 0

        while current_length < target_chars:
            # Randomly select a template
            template = random.choice(self.BACKGROUND_TEMPLATES)
            result.append(template)
            current_length += len(template)

        return "".join(result)[:target_chars]

    def generate_needle(self) -> str:
        """
        Generate a random "needle" (secret information).

        Returns:
            Random alphanumeric string (passkey)
        """
        # Generate random 6-character alphanumeric passkey
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
        Insert needle into context at specified relative position.

        Args:
            context: Background text
            needle: Secret information to insert
            position: Relative position (0.0 = start, 1.0 = end)

        Returns:
            Context with needle inserted
        """
        # Calculate insertion index
        insert_idx = int(len(context) * position)

        # Create needle sentence
        needle_sentence = f" The secret passkey is {needle}. "

        # Insert into context
        result = context[:insert_idx] + needle_sentence + context[insert_idx:]

        return result

    def generate_sample(
        self,
        target_length: int,
        needle_position: float
    ) -> Dict:
        """
        Generate a single NIAH sample.

        Args:
            target_length: Target context length in tokens (rough)
            needle_position: Relative position for needle insertion

        Returns:
            Dictionary with context, question, answer, and metadata
        """
        # Generate background text (4 chars ~= 1 token)
        target_chars = target_length * 4
        background = self.generate_background_text(target_chars)

        # Generate needle
        needle = self.generate_needle()

        # Insert needle
        context = self.insert_needle(background, needle, needle_position)

        # Create sample
        sample = {
            "context": context,
            "question": "What is the secret passkey mentioned in the text?",
            "answer": needle,
            "needle": needle,
            "needle_position": needle_position,
            "context_length_chars": len(context)
        }

        return sample

    def generate_all(self) -> List[Dict]:
        """
        Generate all samples with varying lengths and positions.

        Returns:
            List of NIAH samples
        """
        samples = []

        for i in range(self.num_samples):
            # Vary context length
            min_len, max_len = self.context_length_range
            target_length = random.randint(min_len, max_len)

            # Vary needle position (avoid very start/end)
            needle_position = random.uniform(0.2, 0.8)

            sample = self.generate_sample(target_length, needle_position)
            samples.append(sample)

        return samples

    def save_jsonl(self, samples: List[Dict], filename: str = "niah_samples.jsonl") -> Path:
        """
        Save samples to JSONL file.

        Args:
            samples: List of samples to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        return output_file


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate NIAH evaluation dataset")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/niah",
        help="Output directory"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2000,
        help="Minimum context length (tokens)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="Maximum context length (tokens)"
    )
    parser.add_argument(
        "--output",
        default="niah_samples.jsonl",
        help="Output filename"
    )

    args = parser.parse_args()

    # Create generator
    generator = NIAHGenerator(
        output_dir=args.output_dir,
        num_samples=args.samples,
        context_length_range=(args.min_length, args.max_length)
    )

    print(f"Generating {args.samples} NIAH samples...")
    samples = generator.generate_all()

    print(f"Saving to {args.output_dir}/{args.output}...")
    output_file = generator.save_jsonl(samples, args.output)

    print(f"✅ Generated {len(samples)} samples")
    print(f"📁 Saved to {output_file}")

    # Print statistics
    avg_length = sum(s["context_length_chars"] for s in samples) / len(samples)
    print(f"📊 Average context length: {avg_length:.0f} chars (~{avg_length/4:.0f} tokens)")


if __name__ == "__main__":
    main()
