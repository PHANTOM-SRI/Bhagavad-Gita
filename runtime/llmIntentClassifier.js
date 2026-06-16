/**
 * llmIntentClassifier.js — LLM-powered semantic intent understanding.
 *
 * Replaces the rule-based intentAnalyzer.js with a structured Ollama prompt
 * that understands any natural language query — including indirect, cultural,
 * or emotionally complex phrasing — and returns a canonicalized intent object
 * that the rest of the pipeline (retrieval.js, pipeline.js) can consume.
 *
 * Output shape (identical contract to old analyzeIntent()):
 * {
 *   query_mode          : "emotional" | "informational" | "action"
 *   emotion             : canonical emotion string
 *   emotion_confidence  : 0.0–1.0
 *   emotion_runner_up   : string | null
 *   situation           : short canonical life-situation string
 *   core_struggle       : natural-language synthesis for display
 *   intent_type         : "emotional_support" | "philosophical_seeking" | "action_guidance"
 *   search_bias         : { vectorWeight, emotionWeight, lifeSituationWeight, keywordsWeight }
 *   query_keywords      : string[]
 * }
 */

import fetch from "node-fetch";
import { config as loadEnv } from "dotenv";
import { fileURLToPath } from "url";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
loadEnv({ path: path.resolve(__dirname, "..", ".env") });

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL   = process.env.GEMINI_MODEL   || "gemini-2.5-flash";

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY || "dummy_key_to_allow_init");
const geminiModel = genAI.getGenerativeModel({ model: GEMINI_MODEL });

// ---------------------------------------------------------------------------
// Search bias presets — same as old intentAnalyzer.js (preserves retrieval behaviour)
// ---------------------------------------------------------------------------
const SEARCH_BIAS_PRESETS = {
  emotional_support: {
    vectorWeight: 0.25, emotionWeight: 0.45,
    lifeSituationWeight: 0.25, keywordsWeight: 0.05
  },
  philosophical_seeking: {
    vectorWeight: 0.55, emotionWeight: 0.10,
    lifeSituationWeight: 0.10, keywordsWeight: 0.25
  },
  action_guidance: {
    vectorWeight: 0.35, emotionWeight: 0.20,
    lifeSituationWeight: 0.30, keywordsWeight: 0.15
  }
};

// ---------------------------------------------------------------------------
// Canonical vocabulary lists (passed into the prompt so the LLM stays on-schema)
// ---------------------------------------------------------------------------
const VALID_EMOTIONS = [
  "anxiety", "fear", "grief", "depression", "anger", "envy",
  "greed", "pride", "compassion", "peace", "hope", "clarity",
  "understanding", "seeking", "realization", "neutral",
  "lust", "confusion", "demotivated", "discriminated", "guilt",
  "forgetfulness", "laziness", "loneliness", "hopelessness",
  "forgiveness", "temptation", "uncontrolled mind"
];

const VALID_QUERY_MODES  = ["emotional", "informational", "action"];
const VALID_INTENT_TYPES = ["emotional_support", "philosophical_seeking", "action_guidance"];

// ---------------------------------------------------------------------------
// Lightweight tokenizer for query_keywords (unchanged from old system)
// ---------------------------------------------------------------------------
const STOPWORDS = new Set([
  "i","me","my","myself","we","our","you","your","he","she","it","they","them",
  "their","is","am","are","was","were","be","been","being","have","has","had",
  "do","does","did","will","would","could","should","may","might","must","can",
  "the","a","an","and","but","or","so","if","in","on","at","to","for","of",
  "with","from","by","about","as","into","through","during","that","this",
  "these","those","how","what","why","when","where","who","which","very",
  "just","too","also","both","all","any","not","no","nor","yet","still",
  "than","then","more","most","much","many","some","such","even","ever",
  "never","only","own"
]);

function tokenize(query) {
  return query
    .toLowerCase()
    .replace(/[^a-z0-9\s']/g, " ")
    .split(/\s+/)
    .map(t => t.replace(/^'+|'+$/g, ""))
    .filter(t => t.length > 2 && !STOPWORDS.has(t));
}

// ---------------------------------------------------------------------------
// STAGE 1 — Deterministic pre-screener
// Runs before the LLM. If a strong signal is found, we skip the LLM call.
// Returns a partial core object, or null if the signal is weak/ambiguous.
// ---------------------------------------------------------------------------

// First-person emotional phrasing — very high confidence
const STRONG_EMOTIONAL_PATTERNS = [
  /\bi\s+(feel|am\s+feeling|felt)\b/i,
  /\bi\s+(am|'m)\s+(so\s+)?(anxious|scared|sad|angry|stressed|depressed|worried|lonely|hopeless|broken|lost|overwhelmed|confused|demotivated|discriminated|guilty|sinful|lazy|tempted|jealous|envious|greedy|proud)/i,
  /\bi\s+can'?t\s+(sleep|cope|stop|handle|think|breathe)/i,
  /\bi\s+keep\s+(worrying|crying|thinking about|feeling)/i,
  /\bi\s+(am|feel)\s+anxious/i,
  /\bfeeling\s+(so\s+)?(anxious|scared|sad|angry|stressed|depressed|worried|lonely|hopeless|broken|confused|demotivated|discriminated|guilty|sinful|lazy|tempted|jealous|envious|greedy|proud)/i,
];

// Concrete informational phrasing — very high confidence
const STRONG_INFORMATIONAL_PATTERNS = [
  /^what\s+(is|are|does|did)\b/i,
  /^who\s+(is|was)\b/i,
  /^explain\b/i,
  /^define\b/i,
  /\bwhat\s+does\s+the\s+(gita|bhagavad)/i,
  /\bsummary\s+of\s+chapter/i,
  /\bwhat\s+is\s+(karma|dharma|moksha|yoga|atma)/i,
];

// Emotion keyword map — if the emotional pattern fires, pick the right emotion
const KEYWORD_EMOTION_MAP = [
  { keywords: ["anxious","anxiety","stress","stressed","nervous","overwhelm","worry","worried","exam","exams","test","panic","panicking"], emotion: "anxiety" },
  { keywords: ["afraid","scared","fear","terrif","frightened","dread"], emotion: "fear" },
  { keywords: ["sad","grief","griev","loss","lost someone","passed away","died","mourn","heartbroken","death"], emotion: "grief" },
  { keywords: ["depress","numb","empty","no point","worthless","meaningless","burnout","burned out"], emotion: "depression" },
  { keywords: ["angry","anger","rage","furious","resentment","frustrated","irritated","livid"], emotion: "anger" },
  { keywords: ["lonely","alone","isolated","no one","nobody"], emotion: "loneliness" },
  { keywords: ["guilty","guilt","ashamed","shame","regret","sin","sinful"], emotion: "guilt" },
  { keywords: ["lust","desire","horny","tempted","temptation","addicted"], emotion: "lust" },
  { keywords: ["confused","confusion","lost","don't understand","unsure"], emotion: "confusion" },
  { keywords: ["demotivated","unmotivated","no motivation","give up"], emotion: "demotivated" },
  { keywords: ["discriminate","discriminated","racism","sexism","unfair","injustice"], emotion: "discriminated" },
  { keywords: ["forget","forgetful","forgetfulness","memory"], emotion: "forgetfulness" },
  { keywords: ["lazy","laziness","procrastinate","procrastination"], emotion: "laziness" },
  { keywords: ["envy","envious","jealous","jealousy"], emotion: "envy" },
  { keywords: ["greed","greedy","money","selfish"], emotion: "greed" },
  { keywords: ["pride","proud","ego","arrogant","arrogance"], emotion: "pride" },
  { keywords: ["forgive","forgiveness","resent"], emotion: "forgiveness" },
  { keywords: ["peace","peaceful","calm"], emotion: "peace" },
  { keywords: ["hopeless","hopelessness","losing hope"], emotion: "hopelessness" },
  { keywords: ["uncontrolled mind","distracted","adhd","can't focus","overthinking"], emotion: "uncontrolled mind" },
];

function detectEmotionFromKeywords(q) {
  for (const { keywords, emotion } of KEYWORD_EMOTION_MAP) {
    if (keywords.some(kw => q.includes(kw))) return emotion;
  }
  return null;
}

function preScreen(query) {
  const q = query.toLowerCase();

  // Check for strong informational signal first
  if (STRONG_INFORMATIONAL_PATTERNS.some(re => re.test(query))) {
    return {
      query_mode: "informational",
      emotion: "neutral",
      emotion_confidence: 0.85,
      emotion_runner_up: null,
      situation: "philosophical inquiry",
      core_struggle: "The user is asking an academic or philosophical question about the Bhagavad Gita.",
      intent_type: "philosophical_seeking",
      _prescreened: true
    };
  }

  // Check for strong emotional signal
  if (STRONG_EMOTIONAL_PATTERNS.some(re => re.test(query))) {
    const emotion = detectEmotionFromKeywords(q) || "anxiety";
    // Build a situation hint from context
    let situation = "personal emotional difficulty";
    if (q.includes("exam") || q.includes("test") || q.includes("study")) situation = "exam and academic stress";
    else if (q.includes("job") || q.includes("work") || q.includes("career")) situation = "career and work pressure";
    else if (q.includes("relationship") || q.includes("partner") || q.includes("love")) situation = "relationship difficulty";
    else if (q.includes("family") || q.includes("parent") || q.includes("mother") || q.includes("father")) situation = "family conflict";
    else if (q.includes("death") || q.includes("died") || q.includes("passed away")) situation = "grief from loss";
    else if (q.includes("future") || q.includes("upcoming") || q.includes("tomorrow")) situation = "anxiety about the future";
    return {
      query_mode: "emotional",
      emotion,
      emotion_confidence: 0.88,
      emotion_runner_up: null,
      situation,
      core_struggle: `The person is experiencing ${emotion} related to ${situation} and is seeking support and wisdom from the Gita.`,
      intent_type: "emotional_support",
      _prescreened: true
    };
  }

  return null; // ambiguous — let LLM handle it
}

// ---------------------------------------------------------------------------
// STAGE 2 — Simplified LLM prompt (few-shot, tinyllama-compatible)
// Only called when the pre-screener returns null.
// ---------------------------------------------------------------------------
function buildPrompt(query) {
  return `You are an expert intent classifier. Analyze the user's message and determine the underlying emotion, life situation, and intent. Answer strictly with a valid JSON object.

Rules:
1. "emotion" must be exactly one of the Valid Emotions. If the message is purely philosophical or factual, use "neutral". If they are looking for direction without strong negative feelings, use "seeking".
2. "emotion_runner_up" can be a secondary emotion if they are feeling mixed emotions, or null.
3. "query_mode" is "emotional" for distress/support, "informational" for philosophical questions, and "action" if they need to make a decision or need steps to follow.
4. "intent_type" must be exactly one of the Valid Intents.
5. "situation" should be a 2-4 word summary of their life context (e.g., "career crossroads", "family conflict", "death of a loved one").
6. Map specific concepts to their canonical Valid Emotions: "death of a loved one" -> "grief", "feeling sinful" -> "guilt", "losing hope" -> "hopelessness", "seeking peace" -> "peace", "dealing with envy" -> "envy".

Valid emotions: anxiety, fear, grief, depression, anger, envy, greed, pride, compassion, peace, hope, clarity, understanding, seeking, realization, neutral, lust, confusion, demotivated, discriminated, guilt, forgetfulness, laziness, loneliness, hopelessness, forgiveness, temptation, uncontrolled mind
Valid query_mode: emotional, informational, action
Valid intent_type: emotional_support, philosophical_seeking, action_guidance

Examples:
  Input: 'I am so anxious about my exams'
  Output: {"query_mode":"emotional","emotion":"anxiety","emotion_confidence":0.95,"emotion_runner_up":null,"situation":"exam stress","core_struggle":"The user is overwhelmed and anxious about their upcoming exams.","intent_type":"emotional_support"}

  Input: 'What is karma in the Bhagavad Gita?'
  Output: {"query_mode":"informational","emotion":"neutral","emotion_confidence":0.90,"emotion_runner_up":null,"situation":"philosophical inquiry","core_struggle":"The user is seeking an intellectual understanding of karma.","intent_type":"philosophical_seeking"}

  Input: 'I lost my job and feel hopeless'
  Output: {"query_mode":"emotional","emotion":"depression","emotion_confidence":0.85,"emotion_runner_up":"grief","situation":"job loss","core_struggle":"The user recently lost their job and is struggling with deep feelings of hopelessness.","intent_type":"emotional_support"}

  Input: 'I have two job offers and don't know which one to pick.'
  Output: {"query_mode":"action","emotion":"seeking","emotion_confidence":0.75,"emotion_runner_up":"anxiety","situation":"career crossroads","core_struggle":"The user is facing a difficult career decision and needs guidance on how to choose.","intent_type":"action_guidance"}

  Input: 'My brother betrayed my trust and I can't forgive him.'
  Output: {"query_mode":"emotional","emotion":"anger","emotion_confidence":0.90,"emotion_runner_up":"grief","situation":"family conflict","core_struggle":"The user is holding onto anger and hurt after a betrayal by a family member.","intent_type":"emotional_support"}

Input: '${query.replace(/'/g, "")}'
Output:`;
}

// ---------------------------------------------------------------------------
// LLM call using Gemini
// ---------------------------------------------------------------------------
async function callGeminiForIntent(prompt) {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is not set.");
  }
  
  try {
    const result = await geminiModel.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        responseMimeType: "application/json",
      }
    });
    const text = result.response.text().trim();
    if (!text) throw new Error("Empty Gemini response");
    return text;
  } catch (err) {
    throw err;
  }
}

// ---------------------------------------------------------------------------
// JSON parsing with fallback extraction
// ---------------------------------------------------------------------------
function tryParseJson(text) {
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  const raw = jsonMatch ? jsonMatch[1] : text;
  try {
    return JSON.parse(raw);
  } catch {
    try {
      const start = raw.indexOf("{");
      const end   = raw.lastIndexOf("}");
      if (start !== -1 && end !== -1) return JSON.parse(raw.slice(start, end + 1));
    } catch { /* fall through */ }
    return null;
  }
}

// ---------------------------------------------------------------------------
// Validate & sanitize LLM output to guarantee schema compliance
// ---------------------------------------------------------------------------
function sanitizeIntentResult(raw, query) {
  const emotion = VALID_EMOTIONS.includes(raw?.emotion) ? raw.emotion : "seeking";
  const emotionRunnerUp = VALID_EMOTIONS.includes(raw?.emotion_runner_up)
    ? raw.emotion_runner_up : null;
  const queryMode  = VALID_QUERY_MODES.includes(raw?.query_mode)   ? raw.query_mode   : "emotional";
  const intentType = VALID_INTENT_TYPES.includes(raw?.intent_type) ? raw.intent_type  : "emotional_support";

  const emotionConfidence = (typeof raw?.emotion_confidence === "number" &&
    raw.emotion_confidence >= 0 && raw.emotion_confidence <= 1)
    ? Math.round(raw.emotion_confidence * 100) / 100
    : 0.6;

  const situation    = (typeof raw?.situation    === "string" && raw.situation.trim())
    ? raw.situation.trim()
    : "search for meaning";
  const coreStruggle = (typeof raw?.core_struggle === "string" && raw.core_struggle.trim())
    ? raw.core_struggle.trim()
    : `Seeking guidance around ${situation}.`;

  return {
    query_mode:          queryMode,
    emotion,
    emotion_confidence:  emotionConfidence,
    emotion_runner_up:   emotionRunnerUp,
    situation,
    core_struggle:       coreStruggle,
    intent_type:         intentType
  };
}

// ---------------------------------------------------------------------------
// Build search_bias from intent_type + emotion_confidence
// ---------------------------------------------------------------------------
function buildSearchBias(intentType, emotionConfidence) {
  const bias = { ...SEARCH_BIAS_PRESETS[intentType] };

  // If high-confidence emotional, push emotion weight further
  if (intentType === "emotional_support" && emotionConfidence > 0.85) {
    bias.emotionWeight    = Math.min(bias.emotionWeight + 0.05, 0.6);
    bias.vectorWeight     = Math.max(bias.vectorWeight  - 0.05, 0.15);
  }

  // Normalize
  const sum = bias.vectorWeight + bias.emotionWeight +
              bias.lifeSituationWeight + bias.keywordsWeight;
  const f = 1.0 / sum;
  bias.vectorWeight        = Math.round(bias.vectorWeight        * f * 1000) / 1000;
  bias.emotionWeight       = Math.round(bias.emotionWeight       * f * 1000) / 1000;
  bias.lifeSituationWeight = Math.round(bias.lifeSituationWeight * f * 1000) / 1000;
  bias.keywordsWeight      = Math.round(
    (1.0 - bias.vectorWeight - bias.emotionWeight - bias.lifeSituationWeight) * 1000
  ) / 1000;

  return bias;
}

// ---------------------------------------------------------------------------
// Rule-based fallback (minimal, for when Ollama is unavailable)
// ---------------------------------------------------------------------------
function ruleFallback(query) {
  const q = query.toLowerCase();
  let emotion = "seeking";
  let queryMode = "informational";
  let intentType = "philosophical_seeking";

  const emotionalKeywords = [
    "feel","anxious","anxiety","stress","sad","angry","grief","lost",
    "scared","afraid","hopeless","depressed","overwhelmed","worried","hurt",
    "lonely","broken","confused","stuck","helpless","numb"
  ];
  const actionKeywords = [
    "should i","how do i","what should","help me decide","what to do",
    "how to","need advice","need guidance","what steps"
  ];

  if (emotionalKeywords.some(w => q.includes(w))) {
    emotion = "anxiety";
    queryMode = "emotional";
    intentType = "emotional_support";
  } else if (actionKeywords.some(w => q.includes(w))) {
    queryMode = "action";
    intentType = "action_guidance";
  }

  return {
    query_mode:         queryMode,
    emotion,
    emotion_confidence: 0.4,
    emotion_runner_up:  null,
    situation:          "search for meaning",
    core_struggle:      `The user is ${queryMode === "emotional" ? "experiencing emotional difficulty" : "seeking guidance"} and looking for wisdom from the Gita.`,
    intent_type:        intentType
  };
}

// ---------------------------------------------------------------------------
// PUBLIC API — drop-in async replacement for analyzeIntent()
// ---------------------------------------------------------------------------
export async function analyzeIntent(query) {
  const raw = String(query || "").trim();
  const queryKeywords = tokenize(raw);

  // Stage 1: Try deterministic pre-screener first (fast, reliable)
  const prescreened = preScreen(raw);
  if (prescreened) {
    console.log("[llmIntent] Pre-screener resolved intent:", prescreened.intent_type, "|", prescreened.emotion);
    const { _prescreened, ...core } = prescreened;
    return { ...core, search_bias: buildSearchBias(core.intent_type, core.emotion_confidence), query_keywords: queryKeywords };
  }

  // Stage 2: Pre-screener was uncertain — call LLM for semantic understanding
  console.log("[llmIntent] Ambiguous query — calling Gemini for semantic classification...");

  let core;
  try {
    const prompt = buildPrompt(raw);
    const responseText = await callGeminiForIntent(prompt);
    const parsed = tryParseJson(responseText);

    if (!parsed) {
      console.warn("[llmIntent] Failed to parse LLM JSON → using rule fallback");
      core = ruleFallback(raw);
    } else {
      core = sanitizeIntentResult(parsed, raw);
    }
  } catch (err) {
    console.warn("[llmIntent] Gemini unavailable:", err.message, "→ using rule fallback");
    core = ruleFallback(raw);
  }

  const searchBias = buildSearchBias(core.intent_type, core.emotion_confidence);

  console.log("[llmIntent]", JSON.stringify({
    query_mode:         core.query_mode,
    emotion:            core.emotion,
    emotion_confidence: core.emotion_confidence,
    situation:          core.situation,
    intent_type:        core.intent_type
  }));

  return {
    ...core,
    search_bias:    searchBias,
    query_keywords: queryKeywords
  };
}
