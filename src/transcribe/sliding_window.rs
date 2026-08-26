//! Sliding-window streaming transcription.
//!
//! Ported from nova-npu's `SlidingWindowTranscriber` (MIT) and adapted to
//! voxtype's [`StreamingTranscriber`] trait. Instead of segmenting audio
//! into small VAD-delimited chunks (which degrades Whisper accuracy), this
//! keeps a rolling buffer of the full recording and re-transcribes the
//! whole window every `interval_s` seconds. Because whisper.cpp / OpenVINO
//! inference is fast relative to speech, this gives the model full acoustic
//! context on every pass.
//!
//! New text is extracted by diffing successive transcriptions against the
//! already-committed text, and only the stable tail delta is emitted. The
//! daemon appends each emitted delta at the cursor, so **events must always
//! be deltas** — never cumulative transcripts (or the cursor receives
//! duplicates). The stable-prefix commit policy below guarantees this.
//!
//! ## Commit policy
//!
//! - **Growing mode** (buffer shorter than `max_buffer_seconds`): commit
//!   the common-prefix words between the previous and current Whisper
//!   outputs, gated by `partial_min_words`, advancing `confirmed_words` so
//!   each emission is strictly-new stable text.
//! - **Sliding mode** (buffer trimmed once it wraps): diff the whole
//!   Whisper output against the already-emitted text and commit only the
//!   common prefix of the current delta vs. the previous delta — a tail
//!   word is only committed once stable across two consecutive passes.
//!
//! The engine wraps any batch [`Transcriber`] (whisper.cpp, OpenVINO GenAI,
//! …), so the same code powers every streaming backend.
//!
//! ## Known limitation: stability gating can stall on one unstable word
//!
//! The two-consecutive-pass stability check (both modes) requires the
//! *entire* unconfirmed tail to match between passes before committing any
//! of it — not just the specific word(s) still in flux. If Whisper keeps
//! re-wording one short stretch differently on every single tick (found
//! live: a phrase flickered between "Still said, okay." / "Still doesn't,
//! okay." / "Still so okay." for 18+ consecutive seconds), nothing after
//! that point can commit either, even though the rest of the tail is
//! already well-formed. This isn't data loss — `final_flush` still catches
//! it all at end-of-recording — but it can look like the transcript has
//! stopped responding for an uncomfortably long stretch while the
//! recording keeps running. Properly fixing this means gating stability
//! per-word (or per-stable-run) instead of on the whole remaining delta;
//! that's a real design change, not a quick patch, and hasn't been done.
//!
//! ## Event mapping
//!
//! Each committed delta is emitted as a [`StreamingEvent::Partial`] (typed
//! live at the cursor) when `type_partials` is true, or a
//! [`StreamingEvent::Final`] (commit-only) when false. On graceful EOF the
//! remaining tail delta is emitted as a `Final`, then [`StreamingEvent::Ended`].
//! Cancellation ends the stream with `Ended` and no flush.

use super::streaming::{StreamHandle, StreamingEvent, StreamingTranscriber};
use super::Transcriber;
use crate::error::TranscribeError;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval, MissedTickBehavior};

/// Whisper silence-hallucination phrases. When Whisper is fed silence or
/// noise it frequently emits one of these; drop the result entirely.
const HALLUCINATION_PATTERNS: &[&str] = &[
    "you",
    "the",
    "i",
    "a",
    "it",
    "is",
    "and",
    "to",
    "thank you",
    "thanks",
    "bye",
    "okay",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe",
    "thank you very much",
    "you're welcome",
    "good night",
    "good bye",
    "see you next time",
    "subtitles by the amara.org community",
];

/// Minimum RMS energy for the whole buffer to be treated as speech.
/// Below this we skip transcription entirely (prevents hallucination).
const MIN_SPEECH_RMS: f32 = 0.005;

/// Tuning knobs for the sliding-window engine. Defaults mirror nova-npu.
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindowConfig {
    /// Re-transcribe the whole buffer every `interval_s` seconds.
    pub interval_s: f64,
    /// Maximum buffered audio before the window starts sliding (drops old
    /// samples). Hard-coded to 29.0 s in nova (Whisper context limit).
    pub max_buffer_s: f32,
    /// Assumed sample rate (16 kHz mono for whisper).
    pub sample_rate: u32,
    /// Skip transcription while whole-buffer RMS is below this.
    pub min_speech_rms: f32,
    /// Minimum buffered audio (seconds) before first transcription.
    pub min_audio_s: f32,
    /// Minimum number of new stable words before committing a delta.
    pub partial_min_words: usize,
    /// Emit committed deltas as `Partial` (typed live at the cursor) when
    /// true; emit them as commit-only `Final` segments when false.
    pub type_partials: bool,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            interval_s: 0.8,
            max_buffer_s: 29.0,
            sample_rate: 16_000,
            min_speech_rms: MIN_SPEECH_RMS,
            min_audio_s: 1.0,
            partial_min_words: 2,
            type_partials: true,
        }
    }
}

/// Sliding-window streaming transcriber that wraps any batch [`Transcriber`].
///
/// Implements both [`Transcriber`] (delegating to the wrapped backend) and
/// [`StreamingTranscriber`] (the live-delta engine). The factory constructs
/// this wrapper when the engine's `streaming` config flag is set.
pub struct SlidingWindowStreamingTranscriber {
    base: Arc<dyn Transcriber>,
    config: SlidingWindowConfig,
}

impl SlidingWindowStreamingTranscriber {
    /// Wrap `base` in the sliding-window streaming engine.
    pub fn new(base: Arc<dyn Transcriber>, config: SlidingWindowConfig) -> Self {
        Self { base, config }
    }
}

impl Transcriber for SlidingWindowStreamingTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        self.base.transcribe(samples)
    }

    fn as_streaming(&self) -> Option<&dyn StreamingTranscriber> {
        Some(self)
    }
}

impl StreamingTranscriber for SlidingWindowStreamingTranscriber {
    fn start_stream(
        &self,
        mut samples_rx: mpsc::Receiver<Vec<f32>>,
    ) -> Result<StreamHandle, TranscribeError> {
        let (events_tx, events_rx) = mpsc::channel::<StreamingEvent>(64);
        let (cancel_tx, mut cancel_rx) = oneshot::channel::<()>();

        let base = Arc::clone(&self.base);
        let config = self.config;

        let task = tokio::task::spawn_blocking(move || -> Result<(), TranscribeError> {
            let runtime = tokio::runtime::Handle::current();
            let mut session = Session::new(base, config);

            // Skip the interval's immediate first tick so we don't
            // re-transcribe an empty buffer.
            let mut ticker = interval(Duration::from_secs_f64(config.interval_s));
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            runtime.block_on(ticker.tick());

            enum Outcome {
                Chunk(Vec<f32>),
                Eof,
                Tick,
                Cancelled,
            }

            loop {
                let outcome = runtime.block_on(async {
                    tokio::select! {
                        chunk = samples_rx.recv() => match chunk {
                            Some(c) => Outcome::Chunk(c),
                            None => Outcome::Eof,
                        },
                        _ = ticker.tick() => Outcome::Tick,
                        _ = &mut cancel_rx => Outcome::Cancelled,
                    }
                });

                match outcome {
                    Outcome::Chunk(chunk) => session.feed(&chunk),
                    Outcome::Cancelled => {
                        // Abort promptly; contract allows ending without flush.
                        let _ = runtime.block_on(events_tx.send(StreamingEvent::Ended));
                        return Ok(());
                    }
                    Outcome::Eof => {
                        // Graceful end: one last flush so no audio is lost.
                        if let Some(tail) = session.final_flush() {
                            let _ = runtime.block_on(events_tx.send(StreamingEvent::Final {
                                text: tail,
                                segment_id: 0,
                            }));
                        }
                        let _ = runtime.block_on(events_tx.send(StreamingEvent::Ended));
                        return Ok(());
                    }
                    Outcome::Tick => match session.on_tick() {
                        Ok(deltas) => {
                            for delta in deltas {
                                let event = if config.type_partials {
                                    StreamingEvent::Partial {
                                        text: delta,
                                        segment_id: 0,
                                    }
                                } else {
                                    StreamingEvent::Final {
                                        text: delta,
                                        segment_id: 0,
                                    }
                                };
                                let _ = runtime.block_on(events_tx.send(event));
                            }
                        }
                        Err(err) => {
                            let _ = runtime.block_on(events_tx.send(StreamingEvent::Error(err)));
                            let _ = runtime.block_on(events_tx.send(StreamingEvent::Ended));
                            return Ok(());
                        }
                    },
                }
            }
        });

        // Map the spawn_blocking JoinHandle to the trait's expected shape.
        let task = tokio::spawn(async move {
            match task.await {
                Ok(r) => r,
                Err(join_err) => Err(TranscribeError::InferenceFailed(format!(
                    "Sliding-window streaming task panicked: {}",
                    join_err
                ))),
            }
        });

        Ok(StreamHandle {
            events: events_rx,
            cancel: cancel_tx,
            task,
        })
    }
}

/// Mutable per-session state for one `start_stream` call.
struct Session {
    base: Arc<dyn Transcriber>,
    config: SlidingWindowConfig,

    // Audio accumulator.
    buffer: Vec<f32>,
    /// True once the buffer wrapped (audio was dropped). Switches diffing
    /// from prefix-stable (growing) to tail-delta (sliding).
    sliding: bool,

    // Diff state.
    /// Committed deltas (text already emitted).
    full_text_parts: Vec<String>,
    /// Last raw Whisper output.
    #[allow(dead_code)]
    prev_whisper: String,
    /// Words of the previous transcription (growing mode).
    last_words: Vec<String>,
    /// Words already confirmed & emitted (growing mode).
    confirmed_words: Vec<String>,
    /// Delta words from the previous pass (sliding mode).
    last_delta_words: Vec<String>,
}

impl Session {
    fn new(base: Arc<dyn Transcriber>, config: SlidingWindowConfig) -> Self {
        Self {
            base,
            config,
            buffer: Vec::new(),
            sliding: false,
            full_text_parts: Vec::new(),
            prev_whisper: String::new(),
            last_words: Vec::new(),
            confirmed_words: Vec::new(),
            last_delta_words: Vec::new(),
        }
    }

    /// Append audio and trim once past `max_buffer_s` (enters sliding mode).
    fn feed(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);

        let max_samples = (self.config.max_buffer_s * self.config.sample_rate as f32) as usize;
        let mut dropped = 0usize;
        if self.buffer.len() > max_samples {
            dropped = self.buffer.len() - max_samples;
            self.buffer.drain(..dropped);
        }
        if dropped > 0 && !self.sliding {
            tracing::debug!("[sliding] Buffer full — entering sliding mode");
            self.sliding = true;
        }
    }

    /// Transcribe the whole buffer, applying the silence + hallucination
    /// gates. Returns `None` when there is nothing worth committing.
    fn transcribe_buffer(&self) -> Result<Option<String>, TranscribeError> {
        if (self.buffer.len() as f32) < self.config.min_audio_s * self.config.sample_rate as f32 {
            return Ok(None);
        }
        if rms(&self.buffer) < self.config.min_speech_rms {
            return Ok(None);
        }
        let infer_start = std::time::Instant::now();
        let text = self.base.transcribe(&self.buffer)?;
        let infer_secs = infer_start.elapsed().as_secs_f32();
        let text = text.trim().to_string();
        tracing::trace!(
            "[sliding] tick transcribe -> {text:?} ({} samples, {:.3}s infer, sliding={})",
            self.buffer.len(),
            infer_secs,
            self.sliding,
        );
        // Inference taking longer than the tick interval means every
        // subsequent tick falls further behind, and incoming audio
        // chunks queued on the bounded channel feeding this task start
        // getting silently dropped (see `try_send` at the capture side)
        // instead of backing up — the recording keeps running but the
        // transcript stalls. Surface this loudly since it's otherwise
        // invisible without trace logging.
        if infer_secs > self.config.interval_s as f32 {
            tracing::warn!(
                "[sliding] inference ({:.2}s) exceeded the tick interval ({:.2}s) — \
                 audio chunks may be getting dropped; buffer at {:.1}s",
                infer_secs,
                self.config.interval_s,
                self.buffer.len() as f32 / self.config.sample_rate as f32,
            );
        }
        if text.is_empty() || is_hallucination(&text) {
            return Ok(None);
        }
        Ok(Some(text))
    }

    /// One interval tick: re-transcribe, diff, and return the newly-committed
    /// stable deltas (at most one per tick).
    fn on_tick(&mut self) -> Result<Vec<String>, TranscribeError> {
        let Some(curr) = self.transcribe_buffer()? else {
            return Ok(Vec::new());
        };
        let curr_words: Vec<String> = curr.split_whitespace().map(str::to_owned).collect();
        if curr_words.is_empty() {
            return Ok(Vec::new());
        }

        let mut deltas = Vec::new();

        if self.sliding {
            // Sliding mode: diff the whole output against what's already
            // been emitted, then only commit the portion of the delta that
            // was also present in the previous delta (stable across two
            // consecutive passes).
            let full_emitted = self.full_text_parts.join(" ");
            let already_emitted = last_n_words(&full_emitted, SLIDING_DIFF_LOOKBACK_WORDS);
            // `_confident`, not `extract_new_text`: no confident anchor
            // this tick means "commit nothing, try again next tick" — see
            // extract_new_text's doc comment for why guessing here (its
            // length-based fallback) risks re-emitting already-committed
            // text as a literal duplicate instead of just being briefly
            // unresponsive.
            let mut delta = extract_new_text_confident(&already_emitted, &curr).unwrap_or_default();
            let mut delta_words: Vec<String> =
                delta.split_whitespace().map(str::to_owned).collect();
            tracing::trace!(
                "[sliding] sliding-diff: already_emitted(tail)={already_emitted:?} delta={delta:?} last_delta_words={:?}",
                self.last_delta_words,
            );

            if !self.last_delta_words.is_empty() && !delta_words.is_empty() {
                let stable_n = common_prefix_len(&self.last_delta_words, &delta_words);
                if stable_n >= self.config.partial_min_words {
                    let new = delta_words[..stable_n].join(" ");
                    let new = dedupe_against_emitted(&self.full_text_parts.join(" "), &new);
                    let last_emitted = self.full_text_parts.last();
                    let not_repeat = last_emitted.map(|s| s != &new).unwrap_or(true);
                    if !new.is_empty() && not_repeat {
                        // `new` is a bare word-join with no leading space.
                        // The daemon types deltas verbatim at the cursor, so
                        // separate it from whatever was already committed.
                        let typed = if self.full_text_parts.is_empty() {
                            new.clone()
                        } else {
                            format!(" {new}")
                        };
                        tracing::debug!("[sliding] COMMIT (sliding mode): {typed:?}");
                        self.full_text_parts.push(new);
                        deltas.push(typed);
                        // Re-diff so last_delta_words reflects the remaining
                        // unconfirmed words only.
                        let full_emitted = self.full_text_parts.join(" ");
                        let already_emitted =
                            last_n_words(&full_emitted, SLIDING_DIFF_LOOKBACK_WORDS);
                        delta =
                            extract_new_text_confident(&already_emitted, &curr).unwrap_or_default();
                        delta_words = delta.split_whitespace().map(str::to_owned).collect();
                    } else if !new.is_empty() {
                        tracing::debug!(
                            "[sliding] would-be commit suppressed as repeat of last: {new:?}"
                        );
                    }
                }
            }
            self.last_delta_words = delta_words;
        } else {
            // Growing buffer: commit the common-prefix words between the
            // previous and current transcriptions, advancing confirmed_words.
            if !self.last_words.is_empty() {
                let stable_n = common_prefix_len(&self.last_words, &curr_words);
                if stable_n > self.confirmed_words.len() {
                    let confirmed_new = &curr_words[self.confirmed_words.len()..stable_n];
                    if confirmed_new.len() >= self.config.partial_min_words {
                        let new = confirmed_new.join(" ");
                        let new = dedupe_against_emitted(&self.full_text_parts.join(" "), &new);
                        let last_emitted = self.full_text_parts.last();
                        let not_repeat = last_emitted.map(|s| s != &new).unwrap_or(true);
                        if !new.is_empty() && not_repeat {
                            // `new` is a bare word-join with no leading space.
                            // The daemon types deltas verbatim at the cursor,
                            // so separate it from whatever was already committed.
                            let typed = if self.full_text_parts.is_empty() {
                                new.clone()
                            } else {
                                format!(" {new}")
                            };
                            tracing::debug!("[sliding] COMMIT (growing mode): {typed:?}");
                            self.full_text_parts.push(new);
                            self.confirmed_words.extend(confirmed_new.iter().cloned());
                            deltas.push(typed);
                        }
                    }
                }
            }
        }

        self.prev_whisper = curr;
        self.last_words = curr_words;
        Ok(deltas)
    }

    /// One last transcription at end-of-recording. Returns the remaining tail
    /// delta (never cumulative text — the daemon already typed the partials).
    fn final_flush(&mut self) -> Option<String> {
        let final_text = match self.transcribe_buffer() {
            Ok(Some(t)) => t,
            _ => return None,
        };
        let full_emitted = self.full_text_parts.join(" ");
        let already_emitted = last_n_words(&full_emitted, SLIDING_DIFF_LOOKBACK_WORDS);
        let delta = extract_new_text(&already_emitted, &final_text);
        let delta = delta.trim().to_string();
        let delta = dedupe_against_emitted(&full_emitted, &delta);
        if delta.is_empty() {
            None
        } else {
            // `delta` has no leading space (see `on_tick`); separate it from
            // whatever was already committed before typing it at the cursor.
            let typed = if self.full_text_parts.is_empty() {
                delta.clone()
            } else {
                format!(" {delta}")
            };
            self.full_text_parts.push(delta);
            Some(typed)
        }
    }
}

// ── Diff helpers (ported 1:1 from nova's sliding_window.py) ──────────────

/// Whisper-tolerant word equality: case-insensitive, ignoring trailing
/// punctuation.
fn word_eq(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        s.trim_end_matches(['.', ',', '!', '?', ';', ':'])
            .to_lowercase()
    };
    norm(a) == norm(b)
}

fn words_match<T: AsRef<str>>(a: &[T], b: &[T]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| word_eq(x.as_ref(), y.as_ref()))
}

fn common_prefix_len<T: AsRef<str>>(a: &[T], b: &[T]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && word_eq(a[i].as_ref(), b[i].as_ref()) {
        i += 1;
    }
    i
}

/// How many trailing words of already-committed text to hand to
/// `extract_new_text` as `prev` in sliding mode. Just enough for anchor
/// matching (strategies 1-3 below) — deliberately NOT the whole session.
///
/// `extract_new_text`'s strategy 4 safety check ("Whisper may have fully
/// rewritten the window") compares `curr_words.len()` against
/// `prev_words.len()` and bails to `""` once curr isn't longer. In sliding
/// mode `curr` is always just one ~29s window's worth of words, but
/// `full_text_parts.join(" ")` is the ENTIRE session's cumulative
/// committed text, growing without bound. Once a long-running session's
/// total committed word count exceeds one window's worth (a couple of
/// minutes of continuous dictation, easily), `prev` becomes permanently
/// longer than `curr` and strategy 4 never fires again — the transcript
/// silently and permanently stops advancing, even though new speech keeps
/// transcribing correctly tick over tick. Found live: a 60s recording
/// stopped committing any new text at all about 15s into sliding mode,
/// with every subsequent tick logging `delta=""` despite the raw
/// transcription clearly growing.
const SLIDING_DIFF_LOOKBACK_WORDS: usize = 20;

/// Trailing `n` words of `text` (or the whole thing if shorter).
fn last_n_words(text: &str, n: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let start = words.len().saturating_sub(n);
    words[start..].join(" ")
}

/// Strip any leading words of `candidate` that restate the tail of
/// `already_emitted` — a defense-in-depth guard applied right before a
/// segment is actually committed, independent of whichever diff produced
/// `candidate` in the first place.
///
/// The per-tick diffs already try hard to avoid proposing a duplicate, but
/// they compare against the *whole* current window (`curr`), which is a
/// much noisier, harder match than comparing the final short candidate
/// segment directly against what's already committed. Found live: a
/// duplicate slipped through right at the growing→sliding mode transition
/// — growing mode committed "...and then it spits it out." via its own
/// word-prefix tracking, then sliding mode's first commit re-included
/// "and then it spits it out." verbatim before its genuinely new tail.
/// The two modes track diffing state independently and can briefly
/// disagree about exactly where the boundary falls; this check catches
/// that regardless of which mode is active or why the disagreement
/// happened, rather than trying to keep both modes' state in perfect sync
/// across the transition.
fn dedupe_against_emitted(already_emitted_full: &str, candidate: &str) -> String {
    let tail = last_n_words(already_emitted_full, SLIDING_DIFF_LOOKBACK_WORDS);
    extract_new_text_confident(&tail, candidate).unwrap_or_else(|| candidate.to_string())
}

/// Return the portion of `curr` that is new compared to `prev`, using only
/// strategies that anchor on a real, verified match — never a guess.
///
/// Strategies, in order:
/// 1. Longest suffix→prefix word overlap (the common case).
/// 2. Verbatim string-prefix check.
/// 3. Greedy forward scan — how far into `curr` the `prev` words reach,
///    trusting the position when ≥50% matched (handles Whisper punctuation
///    / casing rewrites).
///
/// Returns `None` when none of these find a confident anchor — the caller
/// should treat that as "nothing safe to commit this pass", not as "no new
/// text". See `extract_new_text`'s doc comment for why this distinction
/// matters for repeated tick callers.
fn extract_new_text_confident(prev: &str, curr: &str) -> Option<String> {
    if prev.is_empty() {
        return Some(curr.to_string());
    }
    if curr.is_empty() {
        return Some(String::new());
    }

    let prev_words: Vec<&str> = prev.split_whitespace().collect();
    let curr_words: Vec<&str> = curr.split_whitespace().collect();

    if prev_words.is_empty() || curr_words.is_empty() {
        return Some(curr.to_string());
    }

    // 1. Exact suffix→prefix overlap (longest wins).
    let mut best_overlap = 0;
    let max_check = prev_words.len().min(curr_words.len());
    for length in 1..=max_check {
        let suffix = &prev_words[prev_words.len() - length..];
        let prefix = &curr_words[..length];
        if words_match(suffix, prefix) {
            best_overlap = length;
        }
    }
    if best_overlap > 0 {
        return Some(curr_words[best_overlap..].join(" "));
    }

    // 2. Verbatim prefix check.
    if let Some(rest) = curr.strip_prefix(prev) {
        return Some(rest.trim().to_string());
    }

    // 3. Greedy forward scan.
    let mut pi = 0;
    let mut best_ci = 0;
    for (ci, cw) in curr_words.iter().enumerate() {
        if pi >= prev_words.len() {
            break;
        }
        if word_eq(prev_words[pi], cw) {
            pi += 1;
            best_ci = ci + 1;
        }
        // else: skip — Whisper rewrote this word, keep scanning.
    }
    if pi as f32 >= prev_words.len() as f32 * 0.5 {
        return Some(curr_words[best_ci..].join(" "));
    }

    None
}

/// Return the portion of `curr` that is new compared to `prev`.
///
/// Tries [`extract_new_text_confident`]'s real-anchor strategies first,
/// then falls back to a length-based guess: if `curr` is not substantially
/// longer than `prev`, Whisper is refining — no new text; otherwise emit
/// only the tail beyond `prev`'s length.
///
/// **This last-resort guess assumes `curr` = `prev` + some new suffix** —
/// true for a single growing buffer re-transcribed in place, but NOT true
/// for the sliding-window tick loop, where `curr` is an independently
/// re-transcribed rolling window and `prev` is a bounded recent tail of
/// the whole session's committed text (see `SLIDING_DIFF_LOOKBACK_WORDS`).
/// Once the genuinely-overlapping part of that tail ages out of the
/// window, this guess's fixed word-count strip no longer lines up with
/// where new content actually starts, and can re-emit already-committed
/// phrasing as if it were new — found live as literal duplicated
/// sentences after a ~50s continuous recording with several repeated
/// filler phrases. `on_tick`'s sliding-mode branch calls
/// `extract_new_text_confident` directly instead, treating "no confident
/// anchor" as "commit nothing this tick" (self-corrects next tick) rather
/// than risk a wrong guess compounding into visible duplication. This
/// function (with the guess) is still right for `final_flush`: a single
/// best-effort guess at the true end of a recording, with no further
/// ticks to compound the mistake, beats losing the trailing text outright.
fn extract_new_text(prev: &str, curr: &str) -> String {
    if let Some(text) = extract_new_text_confident(prev, curr) {
        return text;
    }

    // 4. Safety: Whisper may have fully rewritten the window.
    let prev_words: Vec<&str> = prev.split_whitespace().collect();
    let curr_words: Vec<&str> = curr.split_whitespace().collect();
    if curr_words.len() <= prev_words.len() + 1 {
        return String::new();
    }
    curr_words[prev_words.len()..].join(" ")
}

/// Whole-buffer RMS (sqrt of mean of squares), float32.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Whisper hallucination check: lowercase, strip trailing `.,!?`, membership.
fn is_hallucination(text: &str) -> bool {
    let norm = text.trim().to_lowercase();
    let norm = norm.trim_end_matches(['.', ',', '!', '?']);
    HALLUCINATION_PATTERNS.contains(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    // ── Pure diff helpers ────────────────────────────────────────────

    #[test]
    fn word_eq_ignores_case_and_trailing_punct() {
        assert!(word_eq("Hello,", "hello"));
        assert!(word_eq("WORLD.", "world"));
        assert!(word_eq("you?", "you"));
        assert!(!word_eq("there", "their"));
    }

    #[test]
    fn common_prefix_len_stops_at_first_divergence() {
        let a = ["hello", "world", "foo"];
        let b = ["hello", "WORLD", "bar"];
        assert_eq!(common_prefix_len(&a, &b), 2);
        let c = ["hello"];
        assert_eq!(common_prefix_len(&a, &c), 1);
    }

    #[test]
    fn extract_new_text_empty_prev_returns_curr() {
        assert_eq!(extract_new_text("", "hello world"), "hello world");
    }

    #[test]
    fn extract_new_text_empty_curr_returns_empty() {
        assert_eq!(extract_new_text("hello", ""), "");
    }

    #[test]
    fn extract_new_text_suffix_prefix_overlap() {
        // "you" overlaps: suffix of prev == prefix of curr.
        assert_eq!(extract_new_text("hello world", "world foo bar"), "foo bar");
        // Longest overlap wins: 2 words overlap.
        assert_eq!(
            extract_new_text("we should deploy now", "deploy now please"),
            "please"
        );
    }

    #[test]
    fn extract_new_text_verbatim_prefix() {
        assert_eq!(extract_new_text("hello", "hello world foo"), "world foo");
    }

    #[test]
    fn extract_new_text_fuzzy_scan() {
        // Whisper rewrote punctuation: prev matches 2/3 words fuzzily.
        assert_eq!(extract_new_text("hello, world", "hello world foo"), "foo");
    }

    #[test]
    fn extract_new_text_refining_emits_nothing() {
        // curr no longer than prev → Whisper is refining; no new text.
        assert_eq!(extract_new_text("hello world foo", "hello world"), "");
        assert_eq!(extract_new_text("a b c", "a b"), "");
    }

    #[test]
    fn extract_new_text_safety_tail() {
        // No overlap, prev unreachable via fuzzy scan, curr much longer.
        assert_eq!(extract_new_text("x y", "a b c d"), "c d");
    }

    #[test]
    fn extract_new_text_confident_declines_the_safety_guess() {
        // Same inputs as extract_new_text_safety_tail: no strategies 1-3
        // anchor succeeds. extract_new_text guesses "c d" (strategy 4);
        // the confident variant refuses to guess at all.
        assert_eq!(extract_new_text_confident("x y", "a b c d"), None);
    }

    #[test]
    fn sliding_mode_duplicate_reproduction_and_fix() {
        // Reproduces the shape of a real duplicate/garbled-text bug found
        // live on a ~55s recording with repeated filler phrasing. Once the
        // genuinely-overlapping tail of `already_emitted` no longer
        // matches curr's start via strategies 1-3 (Whisper re-transcribed
        // that stretch slightly differently this pass), extract_new_text's
        // strategy 4 blindly strips `prev_words.len()` words off curr's
        // front on the assumption curr = prev + new suffix. That
        // assumption is wrong in sliding mode (curr is an independently
        // re-transcribed window, not a continuation of prev), so the cut
        // point is arbitrary and can land inside content curr is restating
        // from `prev` — bleeding a fragment of already-committed text back
        // into the "new" output as a garbled partial repeat.
        let prev = "roses are red violets";
        // curr restates content overlapping `prev` ("sugar is sweet" is
        // new to `prev`, but conceptually the kind of near-repeat that
        // trips this up), worded so strategies 1-3 all fail to anchor:
        // curr's first word ("sky") doesn't appear anywhere in `prev`, so
        // no suffix/prefix overlap or fuzzy-scan match is possible.
        let curr = "sky is blue sugar is sweet and so are you my friend";
        assert_eq!(extract_new_text_confident(prev, curr), None);

        // The bug: extract_new_text's guess strips exactly 4 words
        // (prev_words.len()) off curr's front — an arbitrary cut with no
        // relationship to where curr's genuinely new content starts —
        // leaving "is sweet" as a stray fragment in the output.
        let guessed = extract_new_text(prev, curr);
        assert_eq!(guessed, "is sweet and so are you my friend");

        // The fix: the confident variant refuses to guess when no real
        // anchor is found, so the sliding-mode tick loop commits nothing
        // this pass instead of emitting that garbled fragment — it catches
        // up once Whisper's wording stabilizes enough for a real anchor on
        // a later tick.
    }

    #[test]
    fn dedupe_strips_overlap_at_growing_to_sliding_transition() {
        // Reproduces a real duplicate found live: growing mode committed
        // "...and then it spits it out." via its own word-prefix tracking,
        // then the very first sliding-mode commit re-included "and then it
        // spits it out." verbatim before its genuinely new tail — the two
        // modes track diffing state independently and briefly disagreed
        // about exactly where the boundary fell.
        let already_emitted = "It takes a little bit, bit, and then it spits it out.";
        let candidate =
            "and then it spits it out. Yeah, that's what we like to see. I think this is nice.";
        assert_eq!(
            dedupe_against_emitted(already_emitted, candidate),
            "Yeah, that's what we like to see. I think this is nice."
        );
    }

    #[test]
    fn dedupe_leaves_genuinely_new_text_untouched() {
        let already_emitted = "hello world";
        let candidate = "completely unrelated new content";
        assert_eq!(
            dedupe_against_emitted(already_emitted, candidate),
            candidate
        );
    }

    #[test]
    fn last_n_words_returns_tail_or_whole_string() {
        assert_eq!(last_n_words("a b c d e", 3), "c d e");
        assert_eq!(last_n_words("a b", 3), "a b");
        assert_eq!(last_n_words("", 3), "");
        assert_eq!(last_n_words("a b c", 0), "");
    }

    #[test]
    fn long_session_stall_reproduction_and_fix() {
        // Reproduces the empty-forever bug found live on a 60s recording:
        // once a sliding-mode session's cumulative committed text exceeds
        // one window's worth of words, extract_new_text's strategy-4
        // safety check ("curr not substantially longer than prev") starts
        // comparing curr against the WHOLE session instead of just the
        // recent tail, and permanently returns "" even though curr keeps
        // growing with genuinely new speech every tick.
        let mut committed_history: Vec<&str> = Vec::new();
        for i in 0..80 {
            committed_history.push(if i % 2 == 0 { "word" } else { "other" });
        }
        let full_emitted = committed_history.join(" "); // 80 words, no overlap with curr below

        // curr: a fresh ~29s window's transcript, entirely new content
        // (as if the old committed words have aged out of the window),
        // with no exact overlap with `full_emitted` at all — forcing
        // strategy 4 (the length-based safety net) to be the deciding
        // factor, exactly as happens once repeated/rewritten phrasing
        // defeats strategies 1-3 in practice. Must be longer than
        // SLIDING_DIFF_LOOKBACK_WORDS for strategy 4 to have anything to
        // work with once `prev` is properly bounded.
        let curr = "brand new content the user just said in this window \
                     that should absolutely without question be committed \
                     as a real delta and not silently dropped on the floor";
        let curr_words: Vec<&str> = curr.split_whitespace().collect();
        assert!(curr_words.len() > SLIDING_DIFF_LOOKBACK_WORDS);

        // The bug: diffing against the entire unbounded session history.
        assert_eq!(
            extract_new_text(&full_emitted, curr),
            "",
            "sanity check: this is the exact failure mode observed live"
        );

        // The fix: bound `prev` to a small recent tail before diffing.
        // Strategy 4 then strips only SLIDING_DIFF_LOOKBACK_WORDS words off
        // curr's front (its usual "assume that much was already emitted"
        // heuristic) instead of freezing solid — the point of the fix is
        // that *something* keeps flowing, not that it's a perfect diff.
        let bounded = last_n_words(&full_emitted, SLIDING_DIFF_LOOKBACK_WORDS);
        assert_eq!(
            bounded.split_whitespace().count(),
            SLIDING_DIFF_LOOKBACK_WORDS
        );
        let expected = curr_words[SLIDING_DIFF_LOOKBACK_WORDS..].join(" ");
        assert_eq!(extract_new_text(&bounded, curr), expected);
        assert!(!expected.is_empty());
    }

    #[test]
    fn is_hallucination_matches_and_strips_punct() {
        assert!(is_hallucination("thank you."));
        assert!(is_hallucination("  YOU  "));
        assert!(!is_hallucination("hello world"));
        assert!(!is_hallucination("the quick brown fox"));
    }

    #[test]
    fn rms_measures_energy() {
        assert_eq!(rms(&[]), 0.0);
        let silence = vec![0.0; 100];
        assert_eq!(rms(&silence), 0.0);
        let loud = vec![0.5; 100];
        assert!((rms(&loud) - 0.5).abs() < 1e-6);
    }

    /// Deterministic fake backend: transcript grows with buffer length.
    struct FakeTranscriber;

    impl Transcriber for FakeTranscriber {
        fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
            let secs = samples.len() as f32 / 16_000.0;
            let text = if secs < 1.0 {
                ""
            } else if secs < 2.0 {
                "alpha beta"
            } else if secs < 3.0 {
                "alpha beta gamma"
            } else {
                "alpha beta gamma delta"
            };
            Ok(text.to_string())
        }
    }

    fn loud_samples(secs: f32) -> Vec<f32> {
        let n = (secs * 16_000.0) as usize;
        // 0.05 amplitude tone-ish samples — above MIN_SPEECH_RMS.
        (0..n)
            .map(|i| if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect()
    }

    fn streaming_config() -> SlidingWindowConfig {
        SlidingWindowConfig {
            interval_s: 0.02,
            max_buffer_s: 29.0,
            sample_rate: 16_000,
            min_speech_rms: 0.004,
            min_audio_s: 1.0,
            partial_min_words: 1,
            type_partials: true,
        }
    }

    /// Drive a session: feed ~3s of audio, then EOF, and collect events.
    async fn run_session(config: SlidingWindowConfig) -> Vec<StreamingEvent> {
        let transcriber: Arc<dyn Transcriber> = Arc::new(FakeTranscriber);
        let engine = SlidingWindowStreamingTranscriber::new(transcriber, config);

        let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
        let mut handle = engine.start_stream(rx).expect("start stream");

        // Feed three seconds in 0.5s chunks with small gaps between ticks.
        for _ in 0..6 {
            tx.send(loud_samples(0.5)).await.unwrap();
            tokio::time::sleep(Duration::from_millis(60)).await;
        }
        drop(tx); // graceful EOF

        let mut events = Vec::new();
        while let Some(ev) = handle.events.recv().await {
            events.push(ev);
            if matches!(events.last(), Some(StreamingEvent::Ended)) {
                break;
            }
        }
        handle.task.await.unwrap().expect("task ok");
        events
    }

    fn emitted_text(events: &[StreamingEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|ev| match ev {
                StreamingEvent::Partial { text, .. } | StreamingEvent::Final { text, .. } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn session_emits_stable_deltas_then_final_flush_and_ended() {
        let events = run_session(streaming_config()).await;

        // Must end cleanly.
        assert!(matches!(events.last(), Some(StreamingEvent::Ended)));
        assert!(events.len() >= 3, "expected partials + final + ended");

        let parts = emitted_text(&events);
        // Deltas are typed verbatim at the cursor (see `on_tick`), each
        // carrying its own separating leading space — so plain
        // concatenation, not `join(" ")`, is what the daemon actually
        // produces and what must reproduce the full transcript.
        assert_eq!(parts.concat(), "alpha beta gamma delta");
        // Each delta is strictly new text — no duplicates at boundaries.
        for i in 1..parts.len() {
            let tail = parts[..i].concat();
            assert!(!tail.contains(parts[i].trim()), "repeated text: {parts:?}");
        }
    }

    #[tokio::test]
    async fn commit_only_mode_emits_final_deltas() {
        let mut cfg = streaming_config();
        cfg.type_partials = false;
        let events = run_session(cfg).await;

        let parts = emitted_text(&events);
        assert_eq!(parts.concat(), "alpha beta gamma delta");
        assert!(events
            .iter()
            .all(|ev| !matches!(ev, StreamingEvent::Partial { .. })));
    }

    #[tokio::test]
    async fn silence_only_buffer_emits_no_events() {
        let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
        let engine =
            SlidingWindowStreamingTranscriber::new(Arc::new(FakeTranscriber), streaming_config());
        let mut handle = engine.start_stream(rx).expect("start stream");

        // 3s of near-silence — below the RMS gate.
        tx.send(vec![0.0; 16000]).await.unwrap();
        tx.send(vec![0.0; 16000]).await.unwrap();
        tx.send(vec![0.0; 16000]).await.unwrap();
        drop(tx);

        let mut events = Vec::new();
        while let Some(ev) = handle.events.recv().await {
            events.push(ev);
            if matches!(events.last(), Some(StreamingEvent::Ended)) {
                break;
            }
        }
        assert!(matches!(events.last(), Some(StreamingEvent::Ended)));
        assert_eq!(
            emitted_text(&events).len(),
            0,
            "silence should not transcribe"
        );
        handle.task.await.unwrap().unwrap();
    }
}
