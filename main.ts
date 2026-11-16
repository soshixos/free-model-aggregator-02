// free-model-aggregator-02: one server, two client wires, a real free model
// behind every answer.
//
// The trial gives this server a brief (the path in MAKEFUN_PROMPT_FILE) that
// lists fourteen free OpenRouter models with the house's quality ratings, and
// six prompts, each carrying a unique token the answer must carry back. The
// server:
//
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is
//      there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//      "catalog": <catalog>};
//   3. serves GET /v1/models (answering in the shape of the auth header it
//      got: x-api-key means Anthropic, Authorization means OpenAI), POST
//      /v1/chat/completions and POST /v1/messages, each with a JSON body or
//      with server-sent events;
//   4. answers each prompt by dialling ONE free model through the grader's
//      gateway at $OPENAI_BASE_URL/chat/completions. OPENAI_API_KEY in the
//      sandbox is a dummy the gateway swaps for the OpenRouter key pushed to
//      the competition; the real key never enters the sandbox;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//      says {"done": true}. A process still running when the wall ends scores
//      zero, so never wait for anything else.
//
// `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts; the register body's catalog is that same document, byte for
// byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the
//   model it actually dialled. The house records every gateway call and
//   scores the model that really answered. This server sends the brief's
//   model id in the gateway request and puts that same id back on every
//   answer and every streamed chunk (the provider's own answer spells it
//   without the gateway's `openrouter/` prefix), so its claim is the call it
//   made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// ROUTING (quality is 60% of the score)
//   Candidates are the brief's free models that meet the prompt's
//   requirements, best rating first. For this brief that is the
//   thinking-machines model (1475, the best overall) on every plain or tools
//   prompt, and the super-VL model (1303, the best JSON-capable one) on the
//   two JSON-mode prompts. The best rating for a prompt is the best legal
//   rating, so serving the first candidate earns the full quality for that
//   prompt, and every failover step down the list is the smallest quality
//   loss available.
//
// FAILOVER (10% of the score, and the tie-break is the number of gateway calls)
//   A refusal is real: a provider's 429 or 5xx reaches this server as the
//   provider sent it. This server walks the ranked list: on a refusal it
//   records the trail row {status, code, model}, remembers the model as
//   refused for the rest of the trial, and dials the next-best model that
//   meets the prompt. A model that refused once is never spent another call
//   on in this trial — a free tier that rate-limited a model once will do it
//   again within the minute, and there are thirteen other models. Retrying
//   the same model would only burn calls the tie-break counts and time the
//   client's 30-second wait does not have.
//
//   Two refusals are not worth retrying at all:
//   - HTTP 401 means the pushed key is missing or refused: every model
//     would answer the same way, so the server gives up on the prompt at
//     once and stops dialling for the rest of the trial.
//   - HTTP 400 model_not_allowed means the id is not on the gateway's
//     allowlist: that model is skipped like any other refusal.
//
//   A prompt this server still serves after refusals carries the trail on
//   its response, in the header `x-makefun-upstream-failures`: a JSON array
//   of at most 8 rows, each {"status", "code", "model"}, in the order the
//   calls refused. The header is the only way the failover column can see a
//   refusal this server retried and then served, because the env sees the
//   answers and never the gateway's traffic. A response with no refusal
//   carries no header.
//
// DETERMINISM (the final replays every trial)
//   Every choice here is a function of the brief, the requests and the
//   gateway's answers seen so far: the ranked order, the refused set and
//   the 401 short-circuit all derive from those alone. There is no wall
//   clock and no unseeded randomness anywhere in a request path — no
//   Date.now, no Math.random, no timeout that could cut a call short on one
//   pass and not the other — so a replay that serves the recorded gateway
//   answers back leaves the same state export, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  rankedCandidates,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most models one prompt is dialled on: the trail header holds at most 8 rows. */
const MAX_ATTEMPTS_PER_PROMPT = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models that refused a call in this trial: never dialled again in this trial. */
const refusedModels = new Set<string>();
/** A 401 means the pushed key is missing or refused: no model would answer, so stop dialling. */
let authFailed = false;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One row of the upstream-failures trail: the refusal, in the order the calls refused. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The trail header for a prompt that was served after refusals: at most 8 rows, else no header. */
function failuresHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** The provider's error code from a refused gateway body, or a fallback for the status. */
function errorCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed)) {
      const error = isRecord(parsed.error) ? parsed.error : parsed;
      if (typeof error.code === 'string' && error.code !== '') return error.code;
      if (typeof error.type === 'string' && error.type !== '') return error.type;
    }
  } catch {
    // Not JSON: fall through to the status fallback.
  }
  if (status === 400) return 'bad_request';
  if (status === 401) return 'unauthorized';
  if (status === 403) return 'forbidden';
  if (status === 404) return 'not_found';
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'server_error';
  return `http_${status}`;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/**
 * Dial the ranked candidates for one prompt, in order, until one answers.
 * Returns the successful response with the model that served it and the
 * trail of every refusal before it — or null when no candidate answered,
 * with the trail of what was tried. A refused model is remembered for the
 * trial, so a later prompt never spends a call on it. A 401 aborts at once:
 * the key, not the model, is the problem, and no model would answer.
 */
async function servePrompt(
  candidates: ModelRow[],
  body: Json,
): Promise<{ response: Response; model: string; failures: FailureRow[] } | { response: null; model: null; failures: FailureRow[] }> {
  const failures: FailureRow[] = [];
  if (authFailed) return { response: null, model: null, failures };
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
    if (refusedModels.has(candidate.id)) continue;
    attempts += 1;
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, body);
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      refusedModels.add(candidate.id);
      log(`dial ${candidate.id} failed to reach the gateway: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { response: upstream, model: candidate.id, failures };
    const text = await upstream.text().catch(() => '');
    failures.push({ status: upstream.status, code: errorCode(upstream.status, text), model: candidate.id });
    refusedModels.add(candidate.id);
    log(`dial ${candidate.id} refused HTTP ${upstream.status}`);
    if (upstream.status === 401) {
      authFailed = true;
      break;
    }
  }
  return { response: null, model: null, failures };
}

/** A prompt no model answered, as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: FailureRow[]): void {
  const last = failures[failures.length - 1];
  const detail = last !== undefined ? `the gateway answered HTTP ${last.status} (${last.code})` : 'no free model answered';
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail);
  else sendError(res, wire, 502, 'upstream_error', detail);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankedCandidates(brief, requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const served = await servePrompt(candidates, { ...body, model: 'auto' });
  if (served.response === null || served.model === null) return refused(res, 'openai', served.failures);
  const upstream = served.response;
  const model = served.model;
  const headers = failuresHeader(served.failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(upstream.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankedCandidates(brief, requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const served = await servePrompt(candidates, anthropicToOpenAI(body, 'auto'));
  if (served.response === null || served.model === null) return refused(res, 'anthropic', served.failures);
  const upstream = served.response;
  const model = served.model;
  const headers = failuresHeader(served.failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
