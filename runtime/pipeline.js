import fetch from "node-fetch"; // VERY IMPORTANT
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { getTopMatches, getConnectedVerses } from "./retrieval.js";
import { generatePractice } from "./practiceGenerator.js";
import { analyzeIntent } from "./llmIntentClassifier.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const versesDataPath = path.join(projectRoot, "data", "verses.json");
const purportsDataPath = path.join(projectRoot, "data", "purports.json");

const versesData = JSON.parse(fs.readFileSync(versesDataPath, "utf-8"));
const verseDetailsMap = new Map(
  versesData.map(verse => [`${verse.chapter}-${verse.verse}`, verse])
);

let purportsData = [];
let purportDetailsMap = new Map();
if (fs.existsSync(purportsDataPath)) {
  purportsData = JSON.parse(fs.readFileSync(purportsDataPath, "utf-8"));
  purportDetailsMap = new Map(
    purportsData.map(p => [`${p.chapter}-${p.verse}`, p])
  );
}

const TOP_K_RETRIEVAL = parseInt(process.env.TOP_K_RETRIEVAL || "9", 10);
const TOP_K_FINAL = parseInt(process.env.TOP_K_FINAL || "3", 10);

// emotionMap is used to humanise the emotion label in fallback connection strings
const emotionMap = {
  anxiety: "anxiety",
  anger: "anger",
  depression: "hopelessness",
  grief: "grief",
  fear: "fear",
  seeking: "seeking",
  envy: "envy",
  understanding: "confusion",
  pride: "pride",
  greed: "greed",
  compassion: "compassion",
  peace: "peace",
  hope: "hope",
  lust: "lust",
  confusion: "confusion",
  demotivated: "demotivation",
  discriminated: "injustice",
  guilt: "guilt",
  forgetfulness: "forgetfulness",
  laziness: "laziness",
  loneliness: "loneliness",
  hopelessness: "hopelessness",
  forgiveness: "forgiveness",
  temptation: "temptation",
  "uncontrolled mind": "mental instability"
};
const adviceTemplates = {
  anxiety: "Take one small action today without worrying about results",
  fear: "Name what you fear, then take one small action toward it",
  anger: "Pause and respond calmly instead of reacting instantly",
  grief: "Allow yourself to feel the loss, then do one kind thing for yourself",
  depression: "Pick the smallest possible task and complete it — momentum starts there",
  envy: "Redirect your energy inward: list one strength that is uniquely yours",
  greed: "Identify what you already have that is enough, and act from that place",
  pride: "Seek one honest piece of feedback today and sit with it",
  compassion: "Channel your care into one concrete act of service today",
  peace: "Protect your stillness by removing one unnecessary distraction today",
  hope: "Take one step today that your future self will thank you for",
  clarity: "Write down your clearest insight and act on it within the hour",
  understanding: "Choose one option and take the first step immediately",
  realization: "Anchor this realization with one concrete change in your daily routine",
  seeking: "Ask yourself what you are truly looking for, then take one directed step",
  neutral: "Take one calm step forward",
  lust: "Channel your desires into a constructive pursuit today",
  confusion: "Write down what you know to be true, and start from there",
  demotivated: "Do one small task without expecting any immediate reward",
  discriminated: "Stand firm in your inner worth and perform your duty with dignity",
  guilt: "Acknowledge the mistake, learn the lesson, and take a corrective step",
  forgetfulness: "Pause to center your mind and remember your core purpose",
  laziness: "Break your biggest task into a 5-minute action and start now",
  loneliness: "Reach out to someone or connect with your inner spiritual presence",
  hopelessness: "Focus entirely on today's duty, letting go of the distant future",
  forgiveness: "Release one small resentment today to free your own mind",
  temptation: "Observe the urge without acting on it for the next hour",
  "uncontrolled mind": "Focus on your breathing for two minutes to steady your thoughts"
};

const insightVariants = {
  "detachment": [
    "Focus on your actions instead of worrying about results",
    "Let go of outcome pressure and act with clarity",
    "Act with purpose, detached from the result"
  ],
  "control mind": [
    "Control your thoughts instead of reacting automatically",
    "Train your mind to stay steady under pressure",
    "Observe your mind without being controlled by it"
  ],
  "self-knowledge": [
    "Understand your inner self before making decisions",
    "Clarity comes from knowing yourself deeply",
    "Look inward to find the strength you need"
  ],
  "duty": [
    "Fulfill your responsibilities without attachment to comfort",
    "Perform your duty simply because it is the right thing to do",
    "Stay committed to your path, ignoring distractions"
  ],
  "devotion": [
    "Dedicate your actions to a higher purpose",
    "Focus your heart on the deepest truth as you act",
    "Surrender your personal motives to a higher calling"
  ],
  "surrender": [
    "Release what you cannot control and trust the process",
    "Let go of ego and embrace what unfolds",
    "Accept the present moment completely"
  ],
  "discipline": [
    "Master yourself through focused action",
    "True freedom comes from internal discipline",
    "Stay the course through consistent, mindful practice"
  ],
  "knowledge": [
    "Seek true understanding over temporary feelings",
    "Let wisdom guide you, not fleeting emotions",
    "See things as they are, beyond the surface"
  ],
  "equanimity": [
    "Remain steady in both success and failure",
    "Find peace by accepting both praise and blame",
    "Keep a balanced mind regardless of external events"
  ],
  "action": [
    "Take right action without fear of the outcome",
    "Step forward boldly and do what is needed",
    "Let go of hesitation and act with focus"
  ],
  "faith": [
    "Trust the process even amidst uncertainty",
    "Hold tight to your deeper convictions",
    "Have faith that your right actions will lead the way"
  ]
};

function getFallbackInsight(verse, index) {
  // Verse context differentiation
  const chapter = Number(verse.chapter) || parseInt(String(verse.id || "0").split("-")[0], 10);

  if (chapter === 2) {
    const options = insightVariants["action"];
    return options[index % options.length];
  } else if (chapter === 12) {
    const options = insightVariants["devotion"];
    return options[index % options.length];
  } else if (chapter === 13) {
    const options = insightVariants["self-knowledge"];
    return options[index % options.length];
  }

  // Principle-based insight
  const principles = verse.principles || [];
  let candidateOptions = null;

  for (const p of principles) {
    if (p && insightVariants[p.toLowerCase()]) {
      candidateOptions = insightVariants[p.toLowerCase()];
      break;
    }
    // partial matches
    for (const [key, value] of Object.entries(insightVariants)) {
      if (p.toLowerCase().includes(key)) {
        candidateOptions = value;
        break;
      }
    }
    if (candidateOptions) break;
  }

  if (!candidateOptions) {
    candidateOptions = [
      "Stay steady and act without fear",
      "Focus on your path and keep moving forward",
      "Release anxiety by anchoring yourself in the present",
      "Embrace clarity over confusion with steady action"
    ];
  }

  return candidateOptions[index % candidateOptions.length];
}

// Memory function
const historyPath = path.join(projectRoot, "outputs", "history.json");

function loadHistory() {
  if (fs.existsSync(historyPath)) {
    try {
      return JSON.parse(fs.readFileSync(historyPath, "utf-8"));
    } catch {
      return [];
    }
  }
  return [];
}

async function saveQuery(query, result) {
  try {
    const history = loadHistory();
    // Save only lightweight metadata — NOT the full result — to keep the file small
    history.push({
      query,
      timestamp: new Date().toISOString(),
      query_mode: result.understanding?.query_mode,
      emotion: result.understanding?.emotion,
      situation: result.understanding?.situation,
      intent_type: result.understanding?.intent_type,
      verse_ids: (result.guidance || []).map(g => `${g.chapter}:${g.verse}`)
    });
    if (history.length > 50) history.shift();
    await fs.promises.writeFile(historyPath, JSON.stringify(history, null, 2));
  } catch (writeErr) {
    console.warn("[pipeline] Could not save history:", writeErr.message);
  }
}

/** Append a structured log line for observability (7.1) */
async function appendRequestLog(entry) {
  try {
    const logPath = path.join(projectRoot, "outputs", "requests.log");
    const line = JSON.stringify(entry) + "\n";
    await fs.promises.appendFile(logPath, line);
  } catch { /* non-critical */ }
}

function logStage(stage, details = {}) {
  console.log("[pipeline]", JSON.stringify({ stage, ...details }));
}

function safeTrim(value) {
  return (String(value || "")).trim();
}


function clamp01(value) {
  const num = Number(value) || 0;
  if (num < 0) return 0;
  if (num > 1) return 1;
  return num;
}

function normalizeKgWeight(weight) {
  return clamp01((Number(weight) || 0) / 5);
}

function computeKnowledgeGraphSignal(verseId, emotion, situation) {
  const neighbors = getConnectedVerses(verseId, 1).slice(0, 5);
  if (neighbors.length === 0) {
    return { kg_score: 0, neighbors_sample: [] };
  }

  const scored = neighbors.map(item => {
    const connection = item.connection || {};
    const emotionalSignal = (connection.shared_emotion_tags || []).length;
    const situationalSignal = (connection.shared_life_situations || []).length;
    const philosophicalSignal = (connection.shared_principles || []).length;

    // A simple heuristic based on the presence of emotion/situation vs philosophical
    let typeSignal = philosophicalSignal;
    if (emotion !== "seeking") typeSignal += emotionalSignal;
    if (situation !== "general") typeSignal += situationalSignal;

    const edgeStrength = normalizeKgWeight(connection.weight || 0);
    const typeBoost = clamp01(typeSignal / 3);
    const localScore = clamp01((edgeStrength * 0.7) + (typeBoost * 0.3));

    return {
      id: item.id,
      score: localScore,
      weight: connection.weight || 0
    };
  });

  const kgScore = scored.reduce((sum, n) => sum + n.score, 0) / scored.length;
  return {
    kg_score: clamp01(kgScore),
    neighbors_sample: scored.slice(0, 3)
  };
}

function enrichVerse(verse) {
  const details = verseDetailsMap.get(verse.id) || {};
  const purportDetails = purportDetailsMap.get(verse.id) || {};

  const principles = Array.isArray(details.principles) ? details.principles : [];

  // Use purports data if available, else fallback
  const summary = purportDetails.summary || details.summary || "";
  const core_idea = purportDetails.core_idea || details.core_idea || null;

  return {
    ...verse,
    principles,
    summary,
    core_idea
  };
}

function selectTopVersesWithKg(hybridVerses, emotion, situation, topK = 3) {
  const rescored = hybridVerses.map(verse => {
    const baseScore = clamp01(verse.final_score || 0);
    const kgSignal = computeKnowledgeGraphSignal(verse.id, emotion, situation);
    const finalHybridScore = clamp01((baseScore * 0.85) + (kgSignal.kg_score * 0.15));
    const enrichedVerse = enrichVerse(verse);

    return {
      ...enrichedVerse,
      kg_score: kgSignal.kg_score,
      kg_neighbors: kgSignal.neighbors_sample,
      final_hybrid_score: finalHybridScore
    };
  });

  return rescored
    .sort((a, b) => b.final_hybrid_score - a.final_hybrid_score)
    .slice(0, topK);
}

function tryParseJson(text) {
  if (!text) return null;

  // Attempt to extract JSON from markdown blocks if present
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  const rawJson = jsonMatch ? jsonMatch[1] : text;

  try {
    return JSON.parse(rawJson);
  } catch {
    // Attempt aggressive cleanup if it fails
    try {
      const start = rawJson.indexOf("{");
      const end = rawJson.lastIndexOf("}");
      if (start !== -1 && end !== -1) {
        return JSON.parse(rawJson.slice(start, end + 1));
      }
    } catch {
      return null;
    }
  }
  return null;
}

async function callOllama(prompt) {
  logStage("llama_call_request", {
    model: OLLAMA_MODEL,
    base_url: OLLAMA_BASE_URL,
    timeout_ms: OLLAMA_TIMEOUT
  });

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), OLLAMA_TIMEOUT);

  try {
    const res = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt: prompt,
        stream: false,
        format: "json" // try to enforce JSON if supported
      }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      throw new Error(`Ollama API failed (${res.status}): ${await res.text()}`);
    }

    const data = await res.json();
    const text = safeTrim(data?.response);

    if (!text) {
      throw new Error("Empty response from Ollama");
    }

    logStage("llama_call_response", { characters: text.length });
    return text;
  } catch (err) {
    clearTimeout(timeoutId);
    const reason = err.name === "AbortError"
      ? `${OLLAMA_TIMEOUT}ms timeout exceeded`
      : err.message;
    console.error("❌ Ollama Call Error:", reason);
    throw err; // Propagate error to trigger fallback
  }
}

export async function runIntelligentPipeline(query) {
  const startTime = Date.now();
  query = safeTrim(query);
  if (!query) throw new Error("Query is required");

  logStage("start", { query });

  // 1) Analyze query intent via LLM
  const understanding = await analyzeIntent(query);
  logStage("analyze_intent", {
    query_mode: understanding.query_mode,
    emotion: understanding.emotion,
    emotion_confidence: understanding.emotion_confidence,
    situation: understanding.situation,
    intent_type: understanding.intent_type
  });

  // 1b) Compound emotion bias blending (1.3)
  // If runner-up emotion exists and primary confidence is below 0.85,
  // blend search bias 70/30 to honour secondary emotional signal.
  const biasToUse = { ...understanding.search_bias };
  if (understanding.emotion_runner_up && understanding.emotion_confidence < 0.85) {
    const BIAS_PRESETS = {
      emotional_support: { vectorWeight: 0.2, emotionWeight: 0.5, lifeSituationWeight: 0.25, keywordsWeight: 0.05 },
      philosophical_seeking: { vectorWeight: 0.45, emotionWeight: 0.2, lifeSituationWeight: 0.25, keywordsWeight: 0.10 },
      action_guidance: { vectorWeight: 0.35, emotionWeight: 0.25, lifeSituationWeight: 0.30, keywordsWeight: 0.10 }
    };
    // Use philosophical seeking as proxy for the runner-up
    const runnerBias = BIAS_PRESETS["emotional_support"];
    Object.keys(biasToUse).forEach(k => {
      biasToUse[k] = biasToUse[k] * 0.7 + (runnerBias[k] || 0) * 0.3;
    });
  }

  // 2) Hybrid retrieval
  const intentLabels = { emotion: understanding.emotion, situation: understanding.situation };
  const hybridCandidates = await getTopMatches(query, TOP_K_RETRIEVAL, biasToUse, intentLabels);
  const topVerses = selectTopVersesWithKg(hybridCandidates, understanding.emotion, understanding.situation, TOP_K_FINAL);
  logStage("retrieval", {
    selected: topVerses.map(v => ({ id: v.id, score: v.final_hybrid_score }))
  });

  // 3) Related verses from KG neighbors (3.5)
  const relatedVersesRaw = topVerses.length > 0
    ? getConnectedVerses(topVerses[0].id, 1)
      .slice(0, 4)
      .map(n => {
        const details = verseDetailsMap.get(n.id) || {};
        return {
          id: n.id,
          chapter: n.chapter,
          verse: n.verse,
          translation: details.translation || n.translation || "",
          connection_type: (n.connection?.shared_principles || n.connection?.shared_emotion_tags || [])[0] || "related"
        };
      })
    : [];

  // 4) Session memory: last 2 prior queries/emotions for LLM context (3.4)
  const recentHistory = loadHistory().slice(-2);
  const sessionContext = recentHistory.length > 0
    ? `\nPrevious session context (for awareness only, do not repeat):\n` +
    recentHistory.map(h => `- ${h.query} (emotion: ${h.emotion})`).join("\n")
    : "";

  // ──────────────────────────────────────────────────────────────
  // FORK: informational mode vs emotional/action mode  (1.1)
  // ──────────────────────────────────────────────────────────────
  if (understanding.query_mode === "informational") {
    logStage("mode", { type: "informational" });

    // For informational queries: return verse study view with purport excerpts
    const verseStudy = topVerses.map(v => {
      const purport = purportDetailsMap.get(v.id) || {};
      return {
        chapter: v.chapter,
        verse: v.verse,
        translation: v.translation || "",
        purport_excerpt: purport.summary || v.summary || "",
        core_idea: v.core_idea || "",
        principles: v.principles || []
      };
    });

    const informationalOutput = {
      understanding: {
        query_mode: "informational",
        emotion: understanding.emotion,
        emotion_confidence: understanding.emotion_confidence,
        situation: understanding.situation,
        intent_type: understanding.intent_type
      },
      mode: "informational",
      verses: verseStudy,
      guidance: verseStudy.map((v, i) => ({
        chapter: v.chapter,
        verse: v.verse,
        translation: v.translation,
        insight: v.core_idea || getFallbackInsight(v, i),
        connection: v.purport_excerpt || `Chapter ${v.chapter} explores this teaching in depth.`
      })),
      final_advice: "Take time to study these verses. Wisdom deepens with reflection.",
      related_verses: relatedVersesRaw
    };

    // Non-blocking saves
    saveQuery(query, informationalOutput);
    appendRequestLog({
      ts: new Date().toISOString(), query_mode: "informational",
      emotion: understanding.emotion, latency_ms: Date.now() - startTime
    });
    return informationalOutput;
  }

  // ──────────────────────────────────────────────────────────────
  // EMOTIONAL / ACTION MODE (original pipeline)
  // ──────────────────────────────────────────────────────────────

  // 5) Build LLM prompt
  const emotionNoun = emotionMap[understanding.emotion] || understanding.emotion;
  const prompt = `You are a Bhagavad Gita assistant that gives clear, practical guidance.

                User Query: ${query}
                Emotion: ${understanding.emotion}
                Situation: ${understanding.situation}${sessionContext}

                Verses:
                ${topVerses.map(v => `${v.chapter}:${v.verse} - ${v.translation}`).join("\n")}

                Task:
                - For EACH verse: Give 1 short insight (max 15 words) and 1 connection to user situation
                - Give 1 final practical advice (max 20 words)

                Example Output:
{
  "guidance": [
    {
      "chapter": 2,
      "verse": 47,
      "insight": "Focus on effort, not results",
      "connection": "Your anxiety comes from worrying about future outcomes"
    }
  ],
  "final_advice": "Focus on small actions today, not uncertain future results"
}`;

  // 6) Deterministic fallback (built before LLM call)
  const fallbackGuidanceResult = {
    guidance: topVerses.map((v, index) => {
      const isPositiveOrNeutral = ["peace", "clarity", "hope", "compassion", "realization", "neutral", "seeking"].includes(understanding.emotion);
      const connections = isPositiveOrNeutral
        ? [
          `This guides your seeking by offering a steady perspective.`,
          `This provides a firm foundation for your next step.`,
          `This brings clarity by focusing on what truly matters.`
        ]
        : [
          `This reduces ${emotionNoun} by shifting focus away from future worries.`,
          `This helps you stay grounded instead of overthinking outcomes.`,
          `This calms the mind by focusing on what you can control.`
        ];
      return {
        chapter: v.chapter,
        verse: v.verse,
        translation: v.translation || "",
        insight: getFallbackInsight(v, index),
        connection: connections[index % connections.length]
      };
    }),
    final_advice: adviceTemplates[understanding.emotion] || "Take one calm step forward"
  };

  let guidanceResult = fallbackGuidanceResult;
  let usedFallback = true;

  // 7) LLM enhancement
  try {
    const rawLLMOutput = await callOllama(prompt);
    const parsedLLM = tryParseJson(rawLLMOutput);
    const validVerseIds = new Set(topVerses.map(v => `${v.chapter}-${v.verse}`));

    const isValidGuidance =
      parsedLLM &&
      Array.isArray(parsedLLM.guidance) &&
      parsedLLM.guidance.length > 0 &&
      typeof parsedLLM.final_advice === "string" &&
      parsedLLM.guidance.every(
        g =>
          typeof g.chapter === "number" &&
          typeof g.verse === "number" &&
          typeof g.insight === "string" && g.insight.trim() &&
          typeof g.connection === "string" && g.connection.trim() &&
          validVerseIds.has(`${g.chapter}-${g.verse}`)
      );

    if (isValidGuidance) {
      // Inject translation into LLM guidance (3.1)
      guidanceResult = {
        ...parsedLLM,
        guidance: parsedLLM.guidance.map(g => ({
          ...g,
          translation: (verseDetailsMap.get(`${g.chapter}-${g.verse}`) || {}).translation || ""
        }))
      };
      usedFallback = false;
    } else {
      throw new Error("Invalid or incomplete JSON structure from LLM");
    }
  } catch (err) {
    logStage("llama_fallback_triggered", { reason: err.message });
  }

  // 8) Practice generation
  logStage("practice_generation", { mode: "local" });
  const allPrinciples = topVerses.reduce((acc, v) => acc.concat(v.principles || []), []);
  const allCoreIdeas = topVerses.map(v => v.core_idea).filter(Boolean);

  const practice = generatePractice({
    emotion: understanding.emotion,
    situation: understanding.situation,
    core_struggle: understanding.core_struggle,
    principles: allPrinciples,
    core_idea: allCoreIdeas[0]
  });

  // 9) Final Assembly
  const finalOutput = {
    understanding: {
      query_mode: understanding.query_mode,
      emotion: understanding.emotion,
      emotion_confidence: understanding.emotion_confidence,
      emotion_runner_up: understanding.emotion_runner_up,
      situation: understanding.situation,
      core_struggle: understanding.core_struggle,
      intent_type: understanding.intent_type
    },
    mode: "emotional",
    guidance: guidanceResult.guidance,
    final_advice: guidanceResult.final_advice,
    related_verses: relatedVersesRaw,
    practice
  };

  // Non-blocking saves
  const outputPath = path.join(projectRoot, "outputs", "answers.json");
  if (!fs.existsSync(path.join(projectRoot, "outputs"))) {
    fs.mkdirSync(path.join(projectRoot, "outputs"), { recursive: true });
  }
  fs.promises.writeFile(outputPath, JSON.stringify(finalOutput, null, 2)).catch(() => { });
  saveQuery(query, finalOutput);
  appendRequestLog({
    ts: new Date().toISOString(), query_mode: understanding.query_mode,
    emotion: understanding.emotion, situation: understanding.situation,
    intent_type: understanding.intent_type,
    verses_selected: topVerses.map(v => v.id),
    used_llm: !usedFallback, latency_ms: Date.now() - startTime
  });

  logStage("final_output", { used_fallback: usedFallback, latency_ms: Date.now() - startTime });
  return finalOutput;
}

async function runFromCli() {
  const cliQuery = process.argv.slice(2).join(" ");
  const query = safeTrim(cliQuery) || "I feel anxious about my future and don't know what to do";

  try {
    const result = await runIntelligentPipeline(query);
    console.log("\n[Pipeline Complete] Output:");
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error("[pipeline] fatal error", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  runFromCli();
}