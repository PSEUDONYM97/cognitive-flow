package main

import (
	"regexp"
	"strings"
	"unicode"
)

// TextProcessor applies a 6-pass processing pipeline to transcribed text.
type TextProcessor struct {
	replacements map[string]string
}

func NewTextProcessor(replacements map[string]string) *TextProcessor {
	return &TextProcessor{replacements: replacements}
}

func (tp *TextProcessor) Process(text string) string {
	text = tp.removeHallucinationLoops(text)
	text = tp.removeFillerWords(text)
	text = tp.fixWhisperArtifacts(text)
	text = tp.normalizeCharacters(text)
	text = tp.applyCustomReplacements(text)
	text = tp.convertSpokenPunctuation(text)
	return strings.TrimSpace(text)
}

// Pass 1: Detect and collapse 10+ consecutive repeated words
func (tp *TextProcessor) removeHallucinationLoops(text string) string {
	words := strings.Fields(text)
	if len(words) < 10 {
		return text
	}

	var result []string
	repeatCount := 1

	for i := 1; i < len(words); i++ {
		if strings.EqualFold(words[i], words[i-1]) {
			repeatCount++
		} else {
			if repeatCount >= 10 {
				result = append(result, words[i-1])
			} else {
				for j := 0; j < repeatCount; j++ {
					result = append(result, words[i-repeatCount+j])
				}
			}
			repeatCount = 1
		}
	}

	// Handle last word group
	if repeatCount >= 10 {
		result = append(result, words[len(words)-1])
	} else {
		for j := 0; j < repeatCount; j++ {
			result = append(result, words[len(words)-repeatCount+j])
		}
	}

	return strings.Join(result, " ")
}

// Pass 2: Remove filler words (um, uh, er, etc.)
var fillerWords = map[string]bool{
	"um": true, "uh": true, "uhh": true, "umm": true, "ummm": true, "uhhh": true,
	"er": true, "err": true, "errr": true, "ah": true, "ahh": true, "ahhh": true,
	"hmm": true, "hmmm": true, "hmmmm": true, "mm": true, "mmm": true, "mmmm": true,
}

func (tp *TextProcessor) removeFillerWords(text string) string {
	words := strings.Fields(text)
	var result []string
	for _, w := range words {
		// Strip punctuation for comparison
		clean := strings.TrimFunc(w, func(r rune) bool {
			return unicode.IsPunct(r)
		})
		if !fillerWords[strings.ToLower(clean)] {
			result = append(result, w)
		}
	}
	return strings.Join(result, " ")
}

// Pass 3: Fix Whisper artifacts
var whisperArtifacts = []struct {
	pattern     *regexp.Regexp
	replacement string
}{
	{regexp.MustCompile(`,nd\b`), "command"},
	{regexp.MustCompile(`;ng\b`), "thing"},
	{regexp.MustCompile(`:d\b`), "could"},
}

func (tp *TextProcessor) fixWhisperArtifacts(text string) string {
	for _, a := range whisperArtifacts {
		text = a.pattern.ReplaceAllString(text, a.replacement)
	}
	return text
}

// Pass 4: Normalize fancy Unicode to ASCII
var charReplacements = map[rune]string{
	'\u2018': "'", // Left single quote
	'\u2019': "'", // Right single quote
	'\u201C': "\"", // Left double quote
	'\u201D': "\"", // Right double quote
	'\u2013': "-",  // En dash
	'\u2014': "-",  // Em dash
	'\u2026': "...", // Ellipsis
	'\u00A0': " ",  // Non-breaking space
}

func (tp *TextProcessor) normalizeCharacters(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	for _, r := range text {
		if repl, ok := charReplacements[r]; ok {
			b.WriteString(repl)
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// Pass 5: Apply user-defined word replacements (word-boundary match, case-insensitive)
func (tp *TextProcessor) applyCustomReplacements(text string) string {
	for from, to := range tp.replacements {
		pattern := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(from) + `\b`)
		text = pattern.ReplaceAllString(text, to)
	}
	return text
}

// Pass 6: Convert spoken punctuation to actual punctuation
var spokenPunctuation = []struct {
	spoken string
	symbol string
}{
	{"new paragraph", "\n\n"},
	{"new line", "\n"},
	{"question mark", "?"},
	{"exclamation mark", "!"},
	{"exclamation point", "!"},
	{"open quote", "\""},
	{"close quote", "\""},
	{"open paren", "("},
	{"close paren", ")"},
	{"ellipsis", "..."},
	{"period", "."},
	{"comma", ","},
	{"colon", ":"},
	{"semicolon", ";"},
}

func (tp *TextProcessor) convertSpokenPunctuation(text string) string {
	for _, sp := range spokenPunctuation {
		pattern := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(sp.spoken) + `\b`)
		text = pattern.ReplaceAllString(text, sp.symbol)
	}

	// Clean up spacing around punctuation
	text = regexp.MustCompile(` +([.,;:?!])`).ReplaceAllString(text, "$1")
	text = regexp.MustCompile(`([.,;:?!])(\w)`).ReplaceAllString(text, "$1 $2")

	return text
}

// SanitizeForInjection removes control characters and normalizes text for WM_CHAR posting.
func SanitizeForInjection(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	for _, r := range text {
		switch {
		case r == '\n' || r == '\r' || r == '\t':
			b.WriteRune(r)
		case r < 0x20:
			// Skip control characters
		case r == '`':
			b.WriteRune('\'') // Backtick -> single quote
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}
