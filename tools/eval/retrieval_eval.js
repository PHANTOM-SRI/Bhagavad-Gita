import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { getTopMatches } from '../../runtime/retrieval.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function loadJSON(p) {
  return JSON.parse(fs.readFileSync(p, 'utf-8'));
}

const DATA_PATH = path.join(__dirname, '../../data/verses.json');
const TEST_QUERIES_PATH = path.join(__dirname, './retrieval_test_queries.json');
const OUT_DIR = path.join(__dirname, '../../outputs');
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

const verses = loadJSON(DATA_PATH);
const testQueries = loadJSON(TEST_QUERIES_PATH);

const SAMPLE_SIZE = Number(process.env.EVAL_SAMPLE_SIZE) || testQueries.length;
const TOP_K = Number(process.env.TOP_K_RETRIEVAL) || 9;

function buildTestSet(sampleSize) {
  return testQueries.slice(0, sampleSize).map(item => ({
    query: item.query,
    ground_truth: item.relevant_ids
  }));
}

function precisionAtK(retrievedIds, relevantIds, k) {
  const top = retrievedIds.slice(0, k);
  const found = top.filter(id => relevantIds.includes(id)).length;
  return found / k;
}

function recallAtK(retrievedIds, relevantIds, k) {
  const top = retrievedIds.slice(0, k);
  const found = top.filter(id => relevantIds.includes(id)).length;
  return relevantIds.length === 0 ? 0 : found / relevantIds.length;
}

function reciprocalRank(retrievedIds, relevantIds) {
  for (let i = 0; i < retrievedIds.length; i++) {
    if (relevantIds.includes(retrievedIds[i])) return 1 / (i + 1);
  }
  return 0;
}

function ndcgAtK(retrievedIds, relevantIds, k) {
  if (relevantIds.length === 0) return 0;
  let dcg = 0;
  for (let i = 0; i < Math.min(k, retrievedIds.length); i++) {
    if (relevantIds.includes(retrievedIds[i])) {
      dcg += 1 / Math.log2(i + 2);
    }
  }
  let idcg = 0;
  for (let i = 0; i < Math.min(k, relevantIds.length); i++) {
    idcg += 1 / Math.log2(i + 2);
  }
  return idcg > 0 ? dcg / idcg : 0;
}

function averagePrecision(retrievedIds, relevantIds, k) {
  if (relevantIds.length === 0) return 0;
  let hits = 0;
  let sumPrecisions = 0;
  for (let i = 0; i < Math.min(k, retrievedIds.length); i++) {
    if (relevantIds.includes(retrievedIds[i])) {
      hits++;
      sumPrecisions += hits / (i + 1);
    }
  }
  return sumPrecisions / Math.min(k, relevantIds.length);
}

async function evaluate() {
  const testSet = buildTestSet(SAMPLE_SIZE);

  const perQuery = [];

  let sumP = 0, sumR = 0, sumMRR = 0, sumNDCG = 0, sumAP = 0, hits = 0, totalLatency = 0;

  for (const t of testSet) {
    const start = Date.now();
    let results = [];
    try {
      results = await getTopMatches(t.query, TOP_K);
    } catch (err) {
      console.error('Eval: retrieval error for query', t.query, err.message || err);
      results = [];
    }
    const latency = Date.now() - start;
    totalLatency += latency;

    const retrievedIds = results.map(r => r.id || `${r.chapter}-${r.verse}`);

    const p = precisionAtK(retrievedIds, t.ground_truth, TOP_K);
    const r = recallAtK(retrievedIds, t.ground_truth, TOP_K);
    const rr = reciprocalRank(retrievedIds, t.ground_truth);
    const ndcg = ndcgAtK(retrievedIds, t.ground_truth, TOP_K);
    const ap = averagePrecision(retrievedIds, t.ground_truth, TOP_K);

    if (rr > 0) hits += 1;

    sumP += p; sumR += r; sumMRR += rr; sumNDCG += ndcg; sumAP += ap;

    perQuery.push({ query: t.query, ground_truth: t.ground_truth, retrieved: retrievedIds.slice(0, TOP_K), precision: p, recall: r, rr, ndcg, ap, latency_ms: latency });
  }

  const n = testSet.length || 1;
  const metrics = {
    sample_size: n,
    top_k: TOP_K,
    precision_at_k: sumP / n,
    recall_at_k: sumR / n,
    mrr: sumMRR / n,
    map_at_k: sumAP / n,
    ndcg_at_k: sumNDCG / n,
    hit_rate: hits / n,
    avg_latency_ms: totalLatency / n
  };

  const out = { metrics, per_query: perQuery };

  const outPath = path.join(OUT_DIR, 'retrieval_eval_results.json');
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2), 'utf-8');
  console.log('Evaluation complete — results written to', outPath);
  console.log('Summary:', metrics);
}

evaluate().catch(err => {
  console.error('Evaluation failed:', err);
  process.exit(1);
});
