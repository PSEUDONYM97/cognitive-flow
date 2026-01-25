#!/usr/bin/env python3
"""
Language Fingerprint Generator
Analyzes transcription history to extract linguistic patterns and vocabulary.

CONTEXT: This data is from voice-to-AI interaction (speaking to Claude).
The generator separates:
- AI command patterns (how you instruct the AI)
- Natural speech patterns (your actual linguistic fingerprint)
"""

import json
import re
from collections import Counter
from pathlib import Path
from datetime import datetime

HISTORY_PATH = Path.home() / "AppData/Roaming/CognitiveFlow/history.json"
OUTPUT_PATH = Path(__file__).parent / "language_fingerprint.md"

# Minimum characters for a transcription to be included
# Filters out quick commands like "yes", "go ahead", "check that"
MIN_CHARS = 15

# Patterns that are AI-interaction specific, not natural speech
AI_COMMAND_PATTERNS = {
    # Direct AI commands
    'go ahead', 'go ahead and', 'can you', 'could you', 'i want you to',
    'i need you to', 'please go', 'let me know', 'help me', 'show me',
    'tell me', 'give me', 'make sure', 'make sure you', 'make sure to',
    # AI-specific verbs
    'verify', 'check if', 'confirm', 'pull', 'read the', 'search for',
    'look for', 'find the', 'create a', 'generate', 'build', 'update the',
    # Context-setting for AI
    'i want to', 'we need to', 'we should', 'you should', 'you can',
    'you need to', 'you\'re welcome to', 'feel free to',
    # AI acknowledgment
    'that\'s fine', 'that works', 'perfect', 'sounds good', 'good job',
}

# Words that are AI-command context, filter from core vocabulary
AI_CONTEXT_WORDS = {
    'file', 'code', 'function', 'data', 'system', 'app', 'test', 'build',
    'run', 'check', 'verify', 'update', 'create', 'add', 'remove', 'delete',
    'pull', 'push', 'commit', 'branch', 'merge', 'deploy', 'server', 'api',
    'database', 'config', 'settings', 'option', 'parameter', 'variable',
    'context', 'memory', 'tool', 'agent', 'model', 'prompt', 'response',
}


def load_history():
    """Load transcription history."""
    with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_words(text):
    """Extract words from text, lowercased."""
    # Remove punctuation except apostrophes in contractions
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return text.split()


def extract_phrases(text, n=2):
    """Extract n-grams (phrases) from text."""
    words = extract_words(text)
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def analyze_sentence_starters(texts):
    """Find common ways sentences begin."""
    starters = []
    for text in texts:
        sentences = re.split(r'[.!?]+', text)
        for s in sentences:
            s = s.strip()
            if s:
                words = s.split()[:3]  # First 3 words
                if words:
                    starters.append(' '.join(words).lower())
    return Counter(starters)


def find_filler_patterns(texts):
    """Identify filler words and hedging language."""
    filler_indicators = [
        'um', 'uh', 'like', 'you know', 'i mean', 'basically', 'actually',
        'honestly', 'literally', 'kind of', 'sort of', 'i think', 'i guess',
        'maybe', 'probably', 'just', 'really', 'very', 'so', 'okay', 'alright',
        'well', 'anyway', 'right'
    ]

    all_text = ' '.join(texts).lower()
    counts = {}
    for filler in filler_indicators:
        count = len(re.findall(r'\b' + re.escape(filler) + r'\b', all_text))
        if count > 0:
            counts[filler] = count
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def find_unique_expressions(texts, word_freq):
    """Find distinctive phrases that appear multiple times."""
    # Get bigrams and trigrams
    bigrams = Counter()
    trigrams = Counter()

    for text in texts:
        bigrams.update(extract_phrases(text, 2))
        trigrams.update(extract_phrases(text, 3))

    # Filter to phrases appearing 3+ times
    common_bigrams = {k: v for k, v in bigrams.items() if v >= 3}
    common_trigrams = {k: v for k, v in trigrams.items() if v >= 3}

    return common_bigrams, common_trigrams


def analyze_contractions(texts):
    """Find contraction usage patterns."""
    contraction_map = {
        "i'm": "I am", "i've": "I have", "i'll": "I will", "i'd": "I would/had",
        "you're": "you are", "you've": "you have", "you'll": "you will",
        "we're": "we are", "we've": "we have", "we'll": "we will",
        "they're": "they are", "they've": "they have", "they'll": "they will",
        "it's": "it is/has", "that's": "that is/has", "there's": "there is/has",
        "what's": "what is/has", "who's": "who is/has", "how's": "how is/has",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "couldn't": "could not", "won't": "will not",
        "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "let's": "let us", "gonna": "going to", "wanna": "want to",
        "gotta": "got to", "kinda": "kind of", "sorta": "sort of"
    }

    all_text = ' '.join(texts).lower()
    counts = {}
    for contraction in contraction_map:
        count = len(re.findall(r'\b' + re.escape(contraction) + r'\b', all_text))
        if count > 0:
            counts[contraction] = count

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def analyze_command_patterns(texts):
    """Find imperative/command patterns (common in voice-to-AI interaction)."""
    command_starters = [
        'can you', 'could you', 'please', 'go ahead', 'let me', "let's",
        'i need', 'i want', 'make sure', 'don\'t forget', 'remember to',
        'try to', 'help me', 'show me', 'tell me', 'give me', 'send',
        'pull', 'check', 'verify', 'confirm', 'update', 'add', 'remove',
        'delete', 'create', 'open', 'close', 'save', 'fix', 'find'
    ]

    all_text = ' '.join(texts).lower()
    counts = {}
    for cmd in command_starters:
        count = len(re.findall(r'\b' + re.escape(cmd) + r'\b', all_text))
        if count > 0:
            counts[cmd] = count

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def filter_ai_patterns_from_phrases(phrases_counter):
    """Remove AI-command phrases from a phrase counter."""
    filtered = {}
    for phrase, count in phrases_counter.items():
        # Skip if it matches an AI command pattern
        is_ai_pattern = False
        for ai_pattern in AI_COMMAND_PATTERNS:
            if ai_pattern in phrase or phrase in ai_pattern:
                is_ai_pattern = True
                break
        if not is_ai_pattern:
            filtered[phrase] = count
    return filtered


def filter_ai_words_from_vocab(word_freq):
    """Remove AI-context words from vocabulary."""
    return {w: c for w, c in word_freq.items() if w not in AI_CONTEXT_WORDS}


def analyze_natural_expressions(texts):
    """Find expressions that reflect natural speech, not AI commands."""
    # Phrases that indicate genuine speech patterns
    natural_indicators = [
        # Thinking aloud
        'i think', 'i feel like', 'i mean', 'you know', 'honestly',
        'basically', 'actually', 'obviously', 'clearly',
        # Hedging/uncertainty
        'maybe', 'probably', 'might', 'kind of', 'sort of', 'somewhat',
        'i guess', 'i suppose', 'not sure',
        # Emphasis
        'really', 'very', 'super', 'totally', 'absolutely', 'definitely',
        # Transitions
        'anyway', 'so anyway', 'but anyway', 'moving on', 'that said',
        'on the other hand', 'at the same time',
        # Explanatory
        'because', 'since', 'the thing is', 'the point is', 'what i mean',
        # Conversational flow
        'okay so', 'alright so', 'well', 'i don\'t know',
    ]

    all_text = ' '.join(texts).lower()
    counts = {}
    for expr in natural_indicators:
        count = len(re.findall(r'\b' + re.escape(expr) + r'\b', all_text))
        if count > 0:
            counts[expr] = count

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def calculate_metrics(history):
    """Calculate overall speech metrics."""
    total_words = sum(h['words'] for h in history)
    total_duration = sum(h['duration'] for h in history)
    total_chars = sum(h['chars'] for h in history)

    avg_words_per_utterance = total_words / len(history)
    avg_duration = total_duration / len(history)
    words_per_minute = (total_words / total_duration) * 60 if total_duration > 0 else 0
    avg_word_length = total_chars / total_words if total_words > 0 else 0

    return {
        'total_utterances': len(history),
        'total_words': total_words,
        'total_duration_minutes': round(total_duration / 60, 1),
        'avg_words_per_utterance': round(avg_words_per_utterance, 1),
        'avg_utterance_duration_sec': round(avg_duration, 1),
        'words_per_minute': round(words_per_minute, 1),
        'avg_word_length': round(avg_word_length, 1)
    }


def generate_fingerprint():
    """Generate the complete language fingerprint."""
    print("Loading transcription history...")
    history = load_history()

    # Filter: only include transcriptions above minimum character threshold
    # Short utterances are usually quick commands, not natural speech
    filtered_history = [h for h in history if h.get('chars', 0) >= MIN_CHARS]
    skipped = len(history) - len(filtered_history)

    texts = [h['text'] for h in filtered_history if h.get('text')]
    all_words = []
    for text in texts:
        all_words.extend(extract_words(text))

    print(f"Analyzing {len(texts)} transcriptions, {len(all_words)} total words...")
    print(f"  (Filtered out {skipped} short utterances < {MIN_CHARS} chars)")

    # Word frequency
    word_freq = Counter(all_words)

    # Get date range
    timestamps = [h['timestamp'] for h in history]
    date_range = f"{timestamps[-1][:10]} to {timestamps[0][:10]}"

    # Calculate metrics (use filtered history)
    metrics = calculate_metrics(filtered_history)

    # Analyze patterns
    print("Analyzing linguistic patterns...")
    fillers = find_filler_patterns(texts)
    contractions = analyze_contractions(texts)
    commands = analyze_command_patterns(texts)
    starters = analyze_sentence_starters(texts)
    bigrams, trigrams = find_unique_expressions(texts, word_freq)
    natural_expressions = analyze_natural_expressions(texts)

    # Filter out AI patterns for "true" fingerprint
    filtered_bigrams = filter_ai_patterns_from_phrases(bigrams)
    filtered_trigrams = filter_ai_patterns_from_phrases(trigrams)
    filtered_vocab = filter_ai_words_from_vocab(word_freq)

    # Build the fingerprint document
    print("Generating fingerprint document...")

    output = []
    output.append("# Language Fingerprint")
    output.append("")
    output.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    output.append(f"**Data Range:** {date_range}")
    output.append(f"**Source:** Cognitive Flow voice transcriptions (voice-to-AI interaction)")
    output.append("")
    output.append("> **Note:** This data comes from spoken interaction with an AI assistant.")
    output.append("> Patterns are separated into:")
    output.append("> - **Natural speech patterns** - Your actual linguistic fingerprint")
    output.append("> - **AI interaction patterns** - How you command/instruct AI (context-specific)")
    output.append("")
    output.append("---")
    output.append("")

    # ===========================================
    # PART 1: NATURAL SPEECH FINGERPRINT
    # ===========================================
    output.append("# Part 1: Natural Speech Fingerprint")
    output.append("")
    output.append("*These patterns transfer to general speech and writing.*")
    output.append("")

    # Metrics
    output.append("## Speech Metrics")
    output.append("")
    output.append(f"| Metric | Value |")
    output.append(f"|--------|-------|")
    output.append(f"| Total Utterances | {metrics['total_utterances']:,} |")
    output.append(f"| Total Words | {metrics['total_words']:,} |")
    output.append(f"| Total Recording Time | {metrics['total_duration_minutes']} minutes |")
    output.append(f"| Avg Words per Utterance | {metrics['avg_words_per_utterance']} |")
    output.append(f"| Avg Utterance Duration | {metrics['avg_utterance_duration_sec']}s |")
    output.append(f"| Speaking Rate | {metrics['words_per_minute']} WPM |")
    output.append(f"| Avg Word Length | {metrics['avg_word_length']} chars |")
    output.append("")

    # Discourse markers (the real fingerprint)
    output.append("## Discourse Markers & Thinking Patterns")
    output.append("")
    output.append("How you verbalize thought process, hedge, and emphasize:")
    output.append("")
    output.append("| Expression | Count | Category |")
    output.append("|------------|-------|----------|")
    categories = {
        'i think': 'thinking aloud', 'you know': 'thinking aloud', 'i mean': 'thinking aloud',
        'honestly': 'thinking aloud', 'basically': 'thinking aloud', 'actually': 'clarification',
        'obviously': 'assumption', 'clearly': 'assumption',
        'maybe': 'hedging', 'probably': 'hedging', 'might': 'hedging',
        'kind of': 'hedging', 'sort of': 'hedging', 'i guess': 'hedging', 'not sure': 'hedging',
        'really': 'emphasis', 'very': 'emphasis', 'super': 'emphasis', 'totally': 'emphasis',
        'absolutely': 'emphasis', 'definitely': 'emphasis',
        'anyway': 'transition', 'well': 'transition', "i don't know": 'uncertainty',
        'because': 'explanatory', 'since': 'explanatory', 'the thing is': 'explanatory',
    }
    for expr, count in list(natural_expressions.items())[:25]:
        cat = categories.get(expr, 'flow')
        output.append(f"| {expr} | {count} | {cat} |")
    output.append("")

    # Filler words
    output.append("## Verbal Fillers")
    output.append("")
    output.append("Unconscious speech fillers (these are natural and human):")
    output.append("")
    output.append("| Filler | Count |")
    output.append("|--------|-------|")
    pure_fillers = {k: v for k, v in fillers.items()
                    if k in {'um', 'uh', 'like', 'just', 'so', 'well', 'okay', 'alright', 'right'}}
    for filler, count in sorted(pure_fillers.items(), key=lambda x: -x[1]):
        output.append(f"| {filler} | {count} |")
    output.append("")

    # Contractions
    output.append("## Contraction Preferences")
    output.append("")
    output.append("Shows casual vs. formal speech tendency:")
    output.append("")
    output.append("| Contraction | Count |")
    output.append("|-------------|-------|")
    for contraction, count in list(contractions.items())[:15]:
        output.append(f"| {contraction} | {count} |")
    output.append("")

    # Core vocabulary (filtered)
    output.append("## Core Vocabulary")
    output.append("")
    output.append("Meaningful words (excluding stopwords and AI-context terms):")
    output.append("")

    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'could', 'should', 'may', 'might', 'must', 'shall', 'this', 'that',
                 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
                 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
                 'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'if',
                 'then', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
                 'there', 'all', 'each', 'both', 'few', 'more', 'most', 'other', 'some',
                 'such', 'no', 'not', 'only', 'own', 'same', 's', 't', 'm', 've', 'd',
                 'll', 're', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn',
                 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
                 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'about', 'into', 'over',
                 'after', 'before', 'between', 'under', 'again', 'further', 'once',
                 'up', 'down', 'out', 'off', 'above', 'below'}

    meaningful_words = [(w, c) for w, c in filtered_vocab.items()
                        if w not in stopwords and len(w) > 2]
    meaningful_words = sorted(meaningful_words, key=lambda x: -x[1])[:60]

    output.append("| Word | Count | Word | Count | Word | Count |")
    output.append("|------|-------|------|-------|------|-------|")

    for i in range(0, len(meaningful_words), 3):
        row = []
        for j in range(3):
            if i + j < len(meaningful_words):
                w, c = meaningful_words[i + j]
                row.append(f"| {w} | {c}")
            else:
                row.append("| | ")
        output.append(" ".join(row) + " |")
    output.append("")

    # Natural phrases (filtered)
    output.append("## Characteristic Phrases")
    output.append("")
    output.append("Multi-word patterns (AI-command phrases filtered out):")
    output.append("")

    sorted_bigrams = sorted(filtered_bigrams.items(), key=lambda x: -x[1])[:30]
    sorted_trigrams = sorted(filtered_trigrams.items(), key=lambda x: -x[1])[:20]

    output.append("### Two-word phrases")
    output.append("")
    output.append("| Phrase | Count |")
    output.append("|--------|-------|")
    for phrase, count in sorted_bigrams:
        output.append(f"| {phrase} | {count} |")
    output.append("")

    output.append("### Three-word phrases")
    output.append("")
    output.append("| Phrase | Count |")
    output.append("|--------|-------|")
    for phrase, count in sorted_trigrams:
        output.append(f"| {phrase} | {count} |")
    output.append("")

    # ===========================================
    # PART 2: AI INTERACTION PATTERNS (APPENDIX)
    # ===========================================
    output.append("---")
    output.append("")
    output.append("# Part 2: AI Interaction Patterns (Appendix)")
    output.append("")
    output.append("*These patterns are specific to voice-to-AI interaction and may not transfer to general speech.*")
    output.append("")

    # Command patterns
    output.append("## Command Structures")
    output.append("")
    output.append("How you instruct the AI:")
    output.append("")
    output.append("| Pattern | Count |")
    output.append("|---------|-------|")
    for cmd, count in list(commands.items())[:15]:
        output.append(f"| {cmd} | {count} |")
    output.append("")

    # AI-specific vocabulary
    output.append("## AI-Context Vocabulary")
    output.append("")
    output.append("Technical/task words used when working with AI:")
    output.append("")
    ai_vocab = [(w, word_freq[w]) for w in AI_CONTEXT_WORDS if w in word_freq]
    ai_vocab = sorted(ai_vocab, key=lambda x: -x[1])[:20]
    output.append("| Word | Count |")
    output.append("|------|-------|")
    for w, c in ai_vocab:
        output.append(f"| {w} | {c} |")
    output.append("")

    # ===========================================
    # USAGE NOTES
    # ===========================================
    output.append("---")
    output.append("")
    output.append("## Usage Notes")
    output.append("")
    output.append("### For AI Training/Personalization")
    output.append("")
    output.append("Use **Part 1** patterns to train AI to communicate in your style:")
    output.append("- Match contraction frequency (high = casual)")
    output.append("- Use your discourse markers ('you know', 'basically', etc.)")
    output.append("- Match hedging level (how often you qualify statements)")
    output.append("- Adopt your filler patterns for natural-sounding output")
    output.append("")
    output.append("### Key Style Indicators")
    output.append("")

    # Calculate some style indicators
    total_contractions = sum(contractions.values())
    hedging_count = sum(v for k, v in natural_expressions.items()
                        if k in {'maybe', 'probably', 'kind of', 'sort of', 'i guess', 'might'})
    emphasis_count = sum(v for k, v in natural_expressions.items()
                         if k in {'really', 'very', 'super', 'totally', 'absolutely', 'definitely'})

    formality = "casual" if total_contractions > 100 else "moderate" if total_contractions > 50 else "formal"
    hedging_level = "high" if hedging_count > 50 else "moderate" if hedging_count > 20 else "low"
    emphasis_level = "high" if emphasis_count > 30 else "moderate" if emphasis_count > 15 else "low"

    output.append(f"- **Formality:** {formality} (based on contraction usage)")
    output.append(f"- **Hedging tendency:** {hedging_level} (qualifying statements)")
    output.append(f"- **Emphasis tendency:** {emphasis_level} (intensifiers)")
    output.append(f"- **Speaking rate:** {metrics['words_per_minute']} WPM")
    output.append("")
    output.append("### Limitations")
    output.append("")
    output.append("- Data is from voice-to-AI interaction only")
    output.append("- May not reflect casual conversation or formal writing")
    output.append("- Vocabulary skewed toward technical/task domains")
    output.append(f"- Sample: {len(texts)} utterances ({metrics['total_duration_minutes']} min), filtered from {len(history)} total")
    output.append(f"- Short commands (<{MIN_CHARS} chars) excluded")

    # Write output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"\nFingerprint saved to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate_fingerprint()
