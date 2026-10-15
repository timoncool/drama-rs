//! Prompt rows in Drama 3's reference-bracket-v1 layout (sglang-omni prompt.py).

use anyhow::{anyhow, bail, Result};
use tokenizers::Tokenizer;

const TTS_INSTRUCTION: &str = "convert the provided text to speech";
const REFERENCE_INSTRUCTION: &str = "convert the provided text to speech reference to the following:";
const MESSAGE_END: &str = "<|im_end|>\n";
const GENERATION_START: &str = "<|im_start|>assistant\n<|voice|>";
const STRUCTURAL: [&str; 6] = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"];

/// A voice reference: its transcript and codec codes [num_codebooks][frames].
#[derive(Clone)]
pub struct Reference {
    pub transcript: String,
    pub codes: Vec<Vec<i32>>,
}

fn is_ws(c: char) -> bool {
    matches!(c as u32, 0x9..=0xD | 0x20 | 0x85 | 0xA0 | 0x1680 | 0x2000..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000)
}

/// Length of a `<|speaker:N|>` tag (N in 0..=15, no leading zero) at the start of s.
fn speaker_tag_len(s: &str) -> Option<usize> {
    let rest = s.strip_prefix("<|speaker:")?;
    let end = rest.find("|>")?;
    let n = &rest[..end];
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some("<|speaker:".len() + end + 2)
}

fn validate(text: &str, location: &str) -> Result<()> {
    if text.trim_matches(is_ws).is_empty() {
        bail!("{location}: text must not be empty");
    }
    for tok in STRUCTURAL {
        if text.contains(tok) {
            bail!("{location}: {tok} is written by the prompt builder and must not appear in text");
        }
    }
    let mut at = 0;
    while let Some(i) = text[at..].find("<|speaker") {
        let s = &text[at + i..];
        let ok = speaker_tag_len(s).and_then(|len| s["<|speaker:".len()..len - 2].parse::<u32>().ok().filter(|n| *n <= 15).map(|n| (n, len)));
        match ok {
            Some((n, len)) if s["<|speaker:".len()..len - 2] == n.to_string() => at += i + len,
            _ => bail!("{location}: speaker tags must be written <|speaker:N|> with N in 0..15"),
        }
    }
    Ok(())
}

fn with_speaker_tag(text: &str) -> String {
    if speaker_tag_len(text.trim_start_matches(is_ws)).is_some() {
        text.to_string()
    } else {
        format!("<|speaker:0|> {text}")
    }
}

/// Encode text and references as [rows][1 + num_codebooks] frames, ending in the voice turn.
pub fn build_prompt_rows(text: &str, references: &[Reference], tok: &Tokenizer, num_codebooks: usize, codebook_size: usize) -> Result<Vec<i32>> {
    Ok(build_prompt(text, references, tok, num_codebooks, codebook_size)?.0)
}

/// As `build_prompt_rows`, with the number of rows before the text: the voice part, the same for every text spoken
/// with these references.
pub fn build_prompt(text: &str, references: &[Reference], tok: &Tokenizer, num_codebooks: usize, codebook_size: usize) -> Result<(Vec<i32>, usize)> {
    let sem0 = tok.token_to_id("<|semantic:0|>").ok_or_else(|| anyhow!("tokenizer has no <|semantic:0|>"))? as i32;
    let width = num_codebooks + 1;
    let mut rows: Vec<i32> = Vec::new();
    let push_text = |s: &str, rows: &mut Vec<i32>| -> Result<()> {
        let enc = tok.encode(s, true).map_err(|e| anyhow!("tokenize: {e}"))?;
        for &id in enc.get_ids() {
            rows.push(id as i32);
            rows.extend(std::iter::repeat_n(0, num_codebooks));
        }
        Ok(())
    };

    push_text("<|im_start|>system\n", &mut rows)?;
    push_text(if references.is_empty() { TTS_INSTRUCTION } else { REFERENCE_INSTRUCTION }, &mut rows)?;
    for (ri, r) in references.iter().enumerate() {
        let location = format!("references.{ri}");
        validate(&r.transcript, &location)?;
        if r.codes.len() != num_codebooks || r.codes[0].is_empty() || r.codes.iter().any(|c| c.len() != r.codes[0].len()) {
            bail!("{location}: codes must be [{num_codebooks}][frames]");
        }
        if r.codes.iter().flatten().any(|&c| c < 0 || c as usize >= codebook_size) {
            bail!("{location}: codes must be in [0, {codebook_size})");
        }
        // The speaker tag must directly follow "Text: ".
        let tagged = with_speaker_tag(&r.transcript);
        let body = tagged.trim_start_matches(is_ws);
        let lead = &tagged[..tagged.len() - body.len()];
        let tag_len = speaker_tag_len(body).unwrap();
        let transcript = format!("{}{}{}", &body[..tag_len], lead, &body[tag_len..]);
        push_text(&format!("\n\n[Reference {}]\nText: {transcript}\nSpeech: ", ri + 1), &mut rows)?;
        for t in 0..r.codes[0].len() {
            rows.push(r.codes[0][t] + sem0);
            for cb in &r.codes {
                rows.push(cb[t]);
            }
        }
    }
    validate(text, "input")?;
    push_text(MESSAGE_END, &mut rows)?;
    push_text("<|im_start|>user\n", &mut rows)?;
    let voice_rows = rows.len() / width;
    push_text(&with_speaker_tag(text), &mut rows)?;
    push_text(MESSAGE_END, &mut rows)?;
    push_text(GENERATION_START, &mut rows)?;
    debug_assert_eq!(rows.len() % width, 0);
    Ok((rows, voice_rows))
}

/// The leading `<|speaker:N|>` of `text` (after white space), if any.
pub fn leading_speaker(text: &str) -> Option<u32> {
    let s = text.trim_start_matches(is_ws);
    let len = speaker_tag_len(s)?;
    s["<|speaker:".len()..len - 2].parse().ok()
}

/// Every speaker tag in `text`, in order.
fn speakers(text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = text[at..].find("<|speaker:") {
        let s = &text[at + i..];
        match speaker_tag_len(s) {
            Some(len) => {
                if let Ok(n) = s["<|speaker:".len()..len - 2].parse() {
                    out.push(n);
                }
                at += i + len;
            }
            None => at += i + 1,
        }
    }
    out
}

/// The speaker of every line of `text` when it is all one speaker (untagged text is speaker 0).
pub fn sole_speaker(text: &str) -> Option<u32> {
    let mut sp = speakers(text);
    if leading_speaker(text).is_none() {
        sp.insert(0, 0);
    }
    sp.dedup();
    if sp.len() == 1 { Some(sp[0]) } else { None }
}

/// Split text into sentences at . ! ? (and their CJK forms), the way the reference server's
/// streaming endpoint does, so each is generated on its own. Untagged sentences get the speaker
/// tag in force where they start, so splitting never changes who speaks.
pub fn split_sentences(text: &str) -> Vec<String> {
    const END: &[char] = &['.', '!', '?', '。', '！', '？'];
    const CJK_END: &[char] = &['。', '！', '？'];
    const CLOSE: &[char] = &['"', '\'', ')', ']', '”', '’', '»'];
    let mut raw: Vec<&str> = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let (mut start, mut i) = (0, 0);
    while i < chars.len() {
        let c = chars[i].1;
        if END.contains(&c) {
            // Take the whole run of terminators ("?!", "...") and any closing quotes.
            let mut j = i + 1;
            while j < chars.len() && (END.contains(&chars[j].1) || CLOSE.contains(&chars[j].1)) {
                j += 1;
            }
            let end = chars.get(j).map_or(text.len(), |&(b, _)| b);
            // "..." is a pause inside the sentence, not its end.
            let run = &chars[i..j];
            let ellipsis = run.iter().filter(|&&(_, c)| c == '.').count() >= 2 && run.iter().all(|&(_, c)| c == '.' || CLOSE.contains(&c));
            let at_break = !ellipsis && (j == chars.len() || is_ws(chars[j].1) || chars[j].1 == '<' || CJK_END.contains(&chars[j - 1].1));
            if at_break {
                raw.push(&text[start..end]);
                start = end;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    raw.push(&text[start..]);

    let mut out: Vec<String> = Vec::new();
    let mut speaker: Option<u32> = None;
    for seg in raw {
        let seg = seg.trim_matches(is_ws);
        if seg.is_empty() {
            continue;
        }
        // Punctuation-only pieces ("...") belong to the sentence before them.
        if !seg.chars().any(|c| c.is_alphanumeric()) {
            if let Some(last) = out.last_mut() {
                last.push_str(seg);
                continue;
            }
        }
        let tagged = match (leading_speaker(seg), speaker) {
            (None, Some(n)) => format!("<|speaker:{n}|> {seg}"),
            _ => seg.to_string(),
        };
        if let Some(&n) = speakers(&tagged).last() {
            speaker = Some(n);
        }
        out.push(tagged);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences_and_carries_speakers() {
        assert_eq!(split_sentences("Hello there. How are you?"), vec!["Hello there.", "How are you?"]);
        assert_eq!(split_sentences("Wait... what?! No."), vec!["Wait... what?!", "No."]);
        assert_eq!(split_sentences("It costs 3.50 dollars."), vec!["It costs 3.50 dollars."]);
        assert_eq!(split_sentences("He said \"stop.\" Then left."), vec!["He said \"stop.\"", "Then left."]);
        assert_eq!(
            split_sentences("<|speaker:0|> Where were you? <|speaker:1|> [nervous] Out. Just... out. <|speaker:0|> Out where?"),
            vec!["<|speaker:0|> Where were you?", "<|speaker:1|> [nervous] Out.", "<|speaker:1|> Just... out.", "<|speaker:0|> Out where?"]
        );
        assert_eq!(split_sentences("你好。我很好！"), vec!["你好。", "我很好！"]);
        assert_eq!(split_sentences("No ending"), vec!["No ending"]);
        assert_eq!(split_sentences("And then..."), vec!["And then..."]);
    }

    #[test]
    fn speakers_of_text() {
        assert_eq!(sole_speaker("Hi."), Some(0));
        assert_eq!(sole_speaker("<|speaker:2|> Hi."), Some(2));
        assert_eq!(sole_speaker("<|speaker:2|> Hi. <|speaker:2|> Yo."), Some(2));
        assert_eq!(sole_speaker("Hi <|speaker:1|> yo."), None);
    }
}
